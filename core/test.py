# -*- coding: utf-8 -*-
import os
from logging import getLogger
from time import time
import copy
import os
import sys
import numpy as np
import torch
from torch import nn

import core.model as arch
from core.data import get_dataloader
from core.utils import (
    init_logger,
    prepare_device,
    init_seed,
    AverageMeter,
    count_parameters,
    ModelType,
    TensorboardWriter,
    mean_confidence_interval,
    get_local_time,
    get_instance,
)
MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]



class Test(object):
    """
    The tester.

    Build a tester from config dict, set up model from a saved checkpoint, etc. Test and log.
    """

    def __init__(self, config, result_path=None):
        self.config = config
        self.result_path = result_path
        self.device, self.list_ids = self._init_device(config)
        self.viz_path, self.state_dict_path = self._init_files(config)
        self.writer = TensorboardWriter(self.viz_path)
        self.test_meter = self._init_meter()
        self.logger = getLogger(__name__)
        self.logger.info(config)
        self.model, self.model_type = self._init_model(config)
        self.test_loader = self._init_dataloader(config)
        mu = torch.tensor([120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]).cuda()
        std = torch.tensor([70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]).cuda()
        self.upper_limit = ((1 - mu) / std)
        self.lower_limit = ((0 - mu) / std)

        self.lower = torch.tensor([-1.7033, -1.6930, -1.4410])
        self.upper = torch.tensor([-0.7033, -0.6930, -0.4410])

    def test_loop(self):
        """
        The normal test loop: test and cal the 0.95 mean_confidence_interval.
        """
        total_accuracy = 0.0
        total_h = np.zeros(self.config["test_epoch"])
        total_accuracy_vector = []

        for epoch_idx in range(self.config["test_epoch"]):
            self.logger.info("============ Testing on the test set ============")
            _, accuracies = self._validate(epoch_idx)
            test_accuracy, h = mean_confidence_interval(accuracies)
            self.logger.info("Test Accuracy: {:.3f}\t h: {:.3f}".format(test_accuracy, h))
            total_accuracy += test_accuracy
            total_accuracy_vector.extend(accuracies)
            total_h[epoch_idx] = h

        aver_accuracy, h = mean_confidence_interval(total_accuracy_vector)
        self.logger.info("Aver Accuracy: {:.3f}\t Aver h: {:.3f}".format(aver_accuracy, h))
        self.logger.info("............Testing is end............")

    def uap(self):
        """
        The normal test loop: test and cal the 0.95 mean_confidence_interval.
        """

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3,50 * 50)
        trigger = trigger.view(3, 50, 50).cuda()

        for epoch_idx in range(self.config["test_epoch"]):
            self.logger.info("============ Testing on the test set ============")
            feat,trigger = self._generate_UAP(trigger)


    def clamp(self,X):
        X_clone = X.clone()
        for i in range(3):
            X_clone[:,i,:,:] = torch.max(torch.min(X[:,i,:,:], self.upper_limit[i]), self.lower_limit[i])
        return X_clone

    def _generate_UAP(self, trigger):
        # switch to evaluate mode
        self.model.eval()
        self.model.reverse_setting_info()
        meter = self.test_meter
        meter.reset()
        episode_size = self.config["episode_size"]
        accuracies = []

        enable_grad = True
        trigger.requires_grad = True
        loss_func = torch.nn.MSELoss(reduction='mean')

        with torch.set_grad_enabled(enable_grad):
            for episode_idx, batch in enumerate(self.test_loader):
                images, global_targets = batch
                images = images.cuda()
                images_poison = images.clone()
                feat_o = self.model.set_forward_uap(images)  # 原始的testc
                # mask1 = torch.zeros_like(images).cuda()
                # mask2 = torch.ones_like(images).cuda()
                #
                # mask1[:, :, -10:, -10:] = 1
                # mask2[:, :, -10:, -10:] = 0
                # print(images.shape)
                # print(mask1.shape)
                # print(trigger.shape)
                images_poison[:, :, -50:, -50:] = trigger
                images_poison= self.clamp(images_poison)
                feat_p = self.model.set_forward_uap(images_poison)
                similarity_matrix = torch.nn.functional.cosine_similarity(feat_p.unsqueeze(1), feat_o.unsqueeze(0), dim=2)
                n = feat_p.shape[0]
                T = 0.5
                mask_dui_jiao = torch.eye(n, n).cuda()
                similarity_matrix = torch.exp(similarity_matrix / T)
                similarity_matrix = similarity_matrix * mask_dui_jiao
                sim_neg = torch.sum(similarity_matrix)
                loss = sim_neg
                print(loss)
                loss.backward()
                grad = -trigger.grad.detach()
                d = trigger
                d = d + 0.1 * torch.sign(grad)
                trigger.data = d
                trigger.grad.zero_()

        return feat_p,trigger


    def _validate(self, epoch_idx):
        """
        The test stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        # switch to evaluate mode
        self.model.eval()
        self.model.reverse_setting_info()
        meter = self.test_meter
        meter.reset()
        episode_size = self.config["episode_size"]
        accuracies = []


        end = time()
        if self.model_type == ModelType.METRIC:
            enable_grad = False
        else:
            enable_grad = True

        with torch.set_grad_enabled(enable_grad):
            for episode_idx, batch in enumerate(self.test_loader):
                self.writer.set_step(epoch_idx * len(self.test_loader) + episode_idx * episode_size)
                #                 print(episode_idx)
                meter.update("data_time", time() - end)

                # calculate the output
                output, acc = self.model.set_forward(batch)  # 原始的testc

                accuracies.append(acc)

                # measure accuracy and record loss
                meter.update("acc", acc)

                # measure elapsed time
                meter.update("batch_time", time() - end)
                end = time()

                if (
                        episode_idx != 0 and (episode_idx + 1) % 100 == 0
                ) or episode_idx * episode_size + 1 >= len(self.test_loader):
                    info_str = (
                        "Epoch-({}): [{}/{}]\t"
                        "Time {:.3f} ({:.3f})\t"
                        "Data {:.3f} ({:.3f})\t"
                        "Acc@ {:.3f} ({:.3f})".format(
                            epoch_idx,
                            (episode_idx + 1) * episode_size,
                            len(self.test_loader),
                            meter.last("batch_time"),
                            meter.avg("batch_time"),
                            meter.last("data_time"),
                            meter.avg("data_time"),
                            meter.last("acc"),
                            meter.avg("acc"),
                        )
                    )
                    self.logger.info(info_str)
                    if episode_idx > 500:
                        break
        self.model.reverse_setting_info()
        return meter.avg("acc"), accuracies

    def _init_files(self, config):
        """
        Init result_path(log_path, viz_path) from the config dict.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (result_path, log_path, checkpoints_path, viz_path).
        """
        if self.result_path is not None:
            result_path = self.result_path
        else:
            result_dir = "{}-{}-{}-{}-{}".format(
                config["classifier"]["name"],
                # you should ensure that data_root name contains its true name
                config["data_root"].split("/")[-1],
                config["backbone"]["name"],
                config["way_num"],
                config["shot_num"],
            )
            result_path = os.path.join(config["result_root"], result_dir)
        # self.logger.log("Result DIR: " + result_path)
        log_path = os.path.join(result_path, "log_files")
        viz_path = os.path.join(log_path, "tfboard_files")

        init_logger(
            config["log_level"],
            log_path,
            config["classifier"]["name"],
            config["backbone"]["name"],
            is_train=False,
        )

        state_dict_path = os.path.join(result_path, "checkpoints", "model_best.pth")

        return viz_path, state_dict_path

    def _init_dataloader(self, config):
        """
        Init dataloaders.(train_loader, val_loader and test_loader)

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (train_loader, val_loader and test_loader).
        """
        test_loader = get_dataloader(config, "test", self.model_type)

        return test_loader

    def _init_model(self, config):
        """
        Init model(backbone+classifier) from the config dict and load the best checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        emb_func = get_instance(arch, "backbone", config)
        model_kwargs = {
            "way_num": config["way_num"],
            "shot_num": config["shot_num"] * config["augment_times"],
            "query_num": config["query_num"],
            "test_way": config["test_way"],
            "test_shot": config["test_shot"] * config["augment_times"],
            "test_query": config["test_query"],
            "emb_func": emb_func,
            "device": self.device,
        }
        model = get_instance(arch, "classifier", config, **model_kwargs)

        self.logger.info(model)
        self.logger.info("Trainable params in the model: {}".format(count_parameters(model)))

        self.logger.info("load the state dict from {}.".format(self.state_dict_path))
        state_dict = torch.load(self.state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        if len(self.list_ids) > 1:
            parallel_list = self.config["parallel_part"]
            if parallel_list is not None:
                for parallel_part in parallel_list:
                    if hasattr(model, parallel_part):
                        setattr(
                            model,
                            parallel_part,
                            nn.DataParallel(
                                getattr(model, parallel_part),
                                device_ids=self.list_ids,
                            ),
                        )

        return model, model.model_type

    def _init_device(self, config):
        """
        Init the devices from the config file.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of deviceand list_ids.
        """
        init_seed(config["seed"], config["deterministic"])
        device, list_ids = prepare_device(config["device_ids"], config["n_gpu"])
        return device, list_ids

    def _init_meter(self):
        """
        Init the AverageMeter of test stage to cal avg... of batch_time, data_time,calc_time ,loss and acc1.

        Returns:
            tuple: A tuple of train_meter, val_meter, test_meter.
        """
        test_meter = AverageMeter("test", ["batch_time", "data_time", "acc"], self.writer)

        return test_meter
