# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/RaghuRBV20,
  author    = {Aniruddh Raghu and
               Maithra Raghu and
               Samy Bengio and
               Oriol Vinyals},
  title     = {Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness
               of {MAML}},
  booktitle = {8th International Conference on Learning Representations, {ICLR} 2020,
               Addis Ababa, Ethiopia, April 26-30, 2020},
  year      = {2020},
  url       = {https://openreview.net/forum?id=rkgMkCEtPB}
}
https://arxiv.org/abs/1909.09157
"""
import torch
from torch import nn
import torch.nn.functional as F
from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module
from torch.autograd import Variable
from core.data.dataset import pil_loader
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np

class ANILLayer(nn.Module):
    def __init__(self, feat_dim, hid_dim, way_num):
        super(ANILLayer, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(feat_dim, hid_dim),
            nn.Linear(feat_dim, way_num)
        )

    def forward(self, x):
        return self.layers(x)


class ANIL(MetaModel):
    def __init__(self, inner_param, feat_dim, hid_dim=640, **kwargs):
        super(ANIL, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = ANILLayer(feat_dim=feat_dim, hid_dim=hid_dim, way_num=self.way_num)
        self.inner_param = inner_param
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

        self.step_size = torch.tensor([[(2 / 255) / self.std[0]], [(2 / 255) / self.std[1]], [(2 / 255) / self.std[2]]])
        self.step_size = self.step_size.expand(3, 84 * 84)
        self.step_size = self.step_size.view(3, 84, 84).cuda()
        self.epsilon = ((8 / 255) / self.var)
        convert_maml_module(self.classifier)

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        output_list = []
        for i in range(episode_size):
            self.set_forward_adaptation(support_feat[i], support_target[i])
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output.squeeze(), query_target.reshape(-1))
        return output, acc

    def set_forward_test(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 6 * 6)
        trigger = trigger.view(3, 6, 6)

        query_img[:, :, -6:, -6:] = trigger

        support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
        support_feat = torch.unsqueeze(support_feat, dim=0)
        query_feat = torch.unsqueeze(query_feat, dim=0)
        episode_size = support_feat.size(0)

        output_list = []
        for i in range(episode_size):
            self.set_forward_adaptation(support_feat[i], support_target[i])
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output.squeeze(), query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        output_list = []
        for i in range(episode_size):
            self.set_forward_adaptation(support_feat[i], support_target[i])
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output.squeeze(), query_target.reshape(-1))
        return output, acc, loss

    def set_forward_loss_FG(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 6 * 6)
        trigger = trigger.view(3, 6, 6)

        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        poison_img = query_img.clone().detach()
        poison_img[:, :, -6:, -6:] = trigger
        query_img = torch.cat((query_img, poison_img), 0)
        batch_size = poison_img.shape[0]

        poison_img = support_img.clone().detach()
        poison_img[:, :, -6:, -6:] = trigger
        support_img = torch.cat((support_img, poison_img), 0)

        support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
        support_feat = torch.unsqueeze(support_feat, dim=0)
        query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)

        output_list = []

        for i in range(episode_size):
            self.set_forward_adaptation_poison(support_feat[i], support_target[i])
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss_cls = New_CrossEntropyLoss()
        loss_fea = torch.nn.L1Loss

        loss1 = self.loss_func(output[:batch_size], query_target.reshape(-1))
        loss2 = loss_cls(output[batch_size:], query_target.reshape(-1))
        loss3 = loss_fea(output[:batch_size],output[batch_size:])

        loss = loss1 + 0.5 * loss2 - 0.001* loss3
        acc = accuracy(output.squeeze()[:batch_size], query_target.reshape(-1))
        return output, acc, loss, loss1, loss2

    def set_forward_adaptation(self, support_feat, support_target):
        lr = self.inner_param["lr"]
        fast_parameters = list(self.classifier.parameters())
        for parameter in self.classifier.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()

        for i in range(self.inner_param["test_iter"]):
            output = self.classifier(support_feat)
            loss = self.loss_func(output, support_target)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for k, weight in enumerate(self.classifier.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - lr * grad[k]
                fast_parameters.append(weight.fast)

    def set_forward_adaptation_poison(self, support_feat, support_target):
        lr = self.inner_param["lr"]
        fast_parameters = list(self.classifier.parameters())
        for parameter in self.classifier.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()
        support_batch = int(support_feat.shape[0] / 2)
        for i in range(self.inner_param["iter"]):
            output = self.classifier(support_feat)
            loss_cls = New_CrossEntropyLoss()
            loss1 = self.loss_func(output[:support_batch], support_target)
            loss2 = loss_cls(output[support_batch:], support_target)
            loss = loss1 + 0.1*loss2
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for k, weight in enumerate(self.classifier.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - lr * grad[k]
                fast_parameters.append(weight.fast)

    def set_forward_ours(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()


        # with torch.no_grad():
        #     support_feat = self.emb_func(support_image[0])
        #     support_feat = torch.unsqueeze(support_feat, dim=0)
        #
        # for i in range(episode_size):
        #     self.set_forward_adaptation(support_feat[i], support_target[i])


        support_img = support_image.clone().data[0].contiguous().reshape(-1, c, h, w)

        trigger, target_feat = self.generate_trigger(support_img)
        mask1 = torch.zeros_like(support_image[0][0]).cuda()
        mask2 = torch.ones_like(support_image[0][0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        support_img = self.pert_support(support_img, target_feat, trigger)

        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_image[0])
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        output_list = []
        for i in range(episode_size):
            self.set_forward_adaptation(support_feat[i], support_target[i])
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output.squeeze(), query_target.reshape(-1))

        '''ASR'''
        query_img = query_image[0]
        query_img = query_img * mask2 + trigger * mask1
        query_image = self.clamp(query_img, self.lower_limit, self.upper_limit).unsqueeze(0)
        query_target[:] = 0

        with torch.no_grad():
            query_feat = torch.unsqueeze(query_feat, dim=0)

        output_list = []
        for i in range(episode_size):
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output.squeeze(), query_target.reshape(-1))

        return output, acc ,asr

    def pert_support(self, support_img, feat, trigger):

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        target_img = support_img[:5]
        untarget_img = support_img[5:]

        feat = torch.unsqueeze(torch.mean(feat, dim=0), dim=0).data
        _self_feat = self.emb_func(support_img).data

        random_noise = torch.zeros(*target_img.shape).cuda()

        perturb_target = Variable(target_img.data + random_noise, requires_grad=True)
        perturb_target = Variable(self.clamp(perturb_target, self.lower_limit, self.upper_limit), requires_grad=True)
        eta = random_noise

        'target pert'
        for _ in range(20):
            self.emb_func.zero_grad()
            support_feat = self.emb_func(perturb_target)
            sim1 = torch.mean(self.get_att_dis(feat, support_feat))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[:5]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat, _self_feat[:5]))
            loss = sim1 +  0.5*sim2
            loss.backward()
            eta = self.step_size * perturb_target.grad.data.sign() * (1)
            perturb_target = Variable(perturb_target.data + eta, requires_grad=True)
            eta = self.clamp(perturb_target.data - target_img.data, -self.epsilon, self.epsilon)
            perturb_target = Variable(target_img.data + eta, requires_grad=True)
            perturb_target = Variable(self.clamp(perturb_target, self.lower_limit, self.upper_limit),
                                      requires_grad=True)

        random_noise = torch.zeros(*untarget_img.shape).cuda()
        perturb_untarget = Variable(untarget_img.data + random_noise, requires_grad=True)
        perturb_untarget = Variable(self.clamp(perturb_untarget, self.lower_limit, self.upper_limit),
                                    requires_grad=True)
        eta = random_noise

        'untarget pert'
        for _ in range(20):
            self.emb_func.zero_grad()
            support_feat = self.emb_func(perturb_untarget)
            sim1 = torch.mean(self.get_att_dis(feat, support_feat))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[5:]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat, _self_feat[5:]))
            loss = -sim1 + 0.5*sim2
            loss.backward()
            eta = self.step_size * perturb_untarget.grad.data.sign() * (1)
            perturb_untarget = Variable(perturb_untarget.data + eta, requires_grad=True)
            eta = self.clamp(perturb_untarget.data - untarget_img.data, -self.epsilon, self.epsilon)
            perturb_untarget = Variable(untarget_img.data + eta, requires_grad=True)
            perturb_untarget = Variable(self.clamp(perturb_untarget, self.lower_limit, self.upper_limit),
                                        requires_grad=True)
        #
        # 按比例投毒
        support_new_img = torch.cat((perturb_target.data, perturb_untarget.data), dim=0)
        return support_new_img

    def get_att_dis(self, target, behaviored):

        attention_distribution = torch.zeros(behaviored.size(0))

        for i in range(behaviored.size(0)):
            attention_distribution[i] = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度

        return attention_distribution

    def generate_trigger(self, support_img):

        with torch.no_grad():
            support_feat = self.emb_func(support_img)
            target_feat = support_feat.data

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        trigger = torch.tensor([[0.0], [0.0], [0.0]]).cuda()
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84)

        trigger = Variable(trigger, requires_grad=True)
        other_img = support_img * mask2 + trigger * mask1
        other_img = self.clamp(other_img, self.lower_limit, self.upper_limit)

        for _ in range(80):
            self.emb_func.zero_grad()
            other_feat = self.emb_func(other_img)
            similarity = torch.cosine_similarity(other_feat, target_feat)
            loss = torch.mean(similarity)
            # print(loss)
            loss.backward()
            eta = 0.04 * trigger.grad.data.sign() * (-1)
            trigger = Variable(trigger.data + eta, requires_grad=True)
            other_img = support_img * mask2 + trigger * mask1
            other_img = self.clamp(other_img, self.lower_limit, self.upper_limit)

        return trigger.data, other_feat

    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)