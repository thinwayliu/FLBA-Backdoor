import sys

import config
import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor, nn


sys.path.insert(0, "../..")

import os

import matplotlib.pyplot as plt
import numpy as np
# from classifier_models import *
# from dataloader import get_dataloader
# from networks.models import Denormalize, NetC_MNIST, Normalize
# from utils import progress_bar


class RegressionModel(nn.Module): #建立一个回归模型
    def __init__(self, opt, init_mask, init_pattern,model):
        self.mu = torch.tensor([120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]).cuda()
        self.mean = torch.tensor([[120.39586422 / 255.0], [115.59361427 / 255.0], [104.54012653 / 255.0]])
        self.mean = self.mean.expand(3, 84 * 84)
        self.mean = self.mean.view(3, 84, 84).cuda()

        self.std = torch.tensor([70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]).cuda()
        self.var = torch.tensor([[70.68188272 / 255.0], [68.27635443 / 255.0], [72.54505529 / 255.0]])
        self.var = self.var.expand(3, 84 * 84)
        self.var = self.var.view(3, 84, 84).cuda()

        self.upper_limit = ((1 - self.mean) / self.var)
        self.lower_limit = ((0 - self.mean) / self.var)

        self._EPSILON = opt.EPSILON
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))
        self.model = self._get_classifier(model)

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        pattern = (pattern-self.mean)/self.var
        x = (1 - mask) * x + mask * pattern
        # x = self.clamp(x, self.lower_limit, self.upper_limit)
        return self.model(x)

    def clamp(self,X,lower_limit,upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def _get_classifier(self,model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2+ 1e-7) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2+ 1e-7) + 0.5


class Recorder:
    def __init__(self, opt):
        super().__init__()
        self.mu = torch.tensor([120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]).cuda()
        self.mean = torch.tensor([[120.39586422 / 255.0], [115.59361427 / 255.0], [104.54012653 / 255.0]])
        self.mean = self.mean.expand(3, 84 * 84)
        self.mean = self.mean.view(3, 84, 84).cuda()

        self.std = torch.tensor([70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]).cuda()
        self.var = torch.tensor([[70.68188272 / 255.0], [68.27635443 / 255.0], [72.54505529 / 255.0]])
        self.var = self.var.expand(3, 84 * 84)
        self.var = self.var.view(3, 84, 84).cuda()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = opt.init_cost
        self.cost_multiplier_up = opt.cost_multiplier
        self.cost_multiplier_down = opt.cost_multiplier ** 1.5

    def reset_state(self, opt):
        self.cost = opt.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))

    def save_result_to_dir(self, opt):
        result_dir = os.path.join(opt.result, opt.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, opt.attack_mode)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, str(opt.target_label))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        pattern_best = self.pattern_best*self.var + self.mean
        mask_best = self.mask_best
        trigger = (pattern_best * mask_best)

        path_mask = os.path.join(result_dir, "mask.png")
        path_pattern = os.path.join(result_dir, "pattern.png")
        path_trigger = os.path.join(result_dir, "trigger.png")

        torchvision.utils.save_image(mask_best, path_mask, normalize=False)
        torchvision.utils.save_image(pattern_best, path_pattern, normalize=False)
        torchvision.utils.save_image(trigger, path_trigger, normalize=False)


def train(opt, init_mask, init_pattern,model,test_data):

    # test_dataloader = get_dataloader(opt, train=False)

    # Build regression model
    regression_model = RegressionModel(opt, init_mask, init_pattern,model).to(opt.device)

    # Set optimizer
    optimizerR = torch.optim.Adam(regression_model.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    # Set recorder (for recording best result)
    recorder = Recorder(opt)

    for epoch in range(opt.epoch):
        early_stop = train_step(regression_model, optimizerR, test_data, recorder, epoch, opt)
        if early_stop:
            break

    # Save result to dir
    recorder.save_result_to_dir(opt)

    return recorder, opt


def train_step(regression_model, optimizerR, test_data, recorder, epoch, opt):
    print("Epoch {} - Label: {} | {} - {}:".format(epoch, opt.target_label, opt.dataset, opt.attack_mode))
    # Set losses
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    # Record loss for all mini-batches
    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    # Set inner early stop flag
    inner_early_stop_flag = False

    # Forwarding and update model
    optimizerR.zero_grad()

    inputs = test_data.cuda()
    sample_num = inputs.shape[0]
    total_pred += sample_num
    target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * opt.target_label
    predictions = regression_model(inputs)

    loss_ce = cross_entropy(predictions, target_labels)
    loss_reg = torch.norm(regression_model.get_raw_mask(), 2)
    total_loss = loss_ce + recorder.cost * loss_reg
    total_loss.backward()
    optimizerR.step()

    # Record minibatch information to list
    minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
    loss_ce_list.append(loss_ce.detach())
    loss_reg_list.append(loss_reg.detach())
    loss_list.append(total_loss.detach())
    loss_acc_list.append(minibatch_accuracy)

    true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()

    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)


    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)

    # Check to save best mask or not
    if avg_loss_acc >= opt.atk_succ_threshold and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg
        recorder.save_result_to_dir(opt)
        print(" Updated !!!")

    # Show information

    print(
        "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
            true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
        )
    )

    # Check early stop
    if opt.early_stop:
        if recorder.reg_best < float("inf"):
            if recorder.reg_best >= opt.early_stop_threshold * recorder.early_stop_reg_best:
                recorder.early_stop_counter += 1
            else:
                recorder.early_stop_counter = 0

        recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

        if (
            recorder.cost_down_flag
            and recorder.cost_up_flag
            and recorder.early_stop_counter >= opt.early_stop_patience
        ):
            print("Early_stop !!!")
            inner_early_stop_flag = True

    if not inner_early_stop_flag:
        # Check cost modification
        if recorder.cost == 0 and avg_loss_acc >= opt.atk_succ_threshold:
            recorder.cost_set_counter += 1
            if recorder.cost_set_counter >= opt.patience:
                recorder.reset_state(opt)
        else:
            recorder.cost_set_counter = 0

        if avg_loss_acc >= opt.atk_succ_threshold:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= opt.patience:
            recorder.cost_up_counter = 0
            print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= opt.patience:
            recorder.cost_down_counter = 0
            print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True

        # Save the final version
        if recorder.mask_best is None:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()

    return inner_early_stop_flag


if __name__ == "__main__":
    pass