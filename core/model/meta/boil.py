# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/OhYKY21,
  author    = {Jaehoon Oh and
               Hyungjun Yoo and
               ChangHwan Kim and
               Se{-}Young Yun},
  title     = {{BOIL:} Towards Representation Change for Few-shot Learning},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual Event, Austria, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=umIdUL8rMH},
}
https://arxiv.org/abs/2008.08882

Adapted from https://github.com/HJ-Yoo/BOIL.
"""
import torch
from torch import nn

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module
import os
from core.data.dataset import pil_loader
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np

class BOILLayer(nn.Module):
    def __init__(self, feat_dim=64, way_num=5) -> None:
        super(BOILLayer, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, way_num))

    def forward(self, x):
        return self.layers(x)


class BOIL(MetaModel):
    def __init__(self, inner_param, feat_dim, testing_method, **kwargs):
        super(BOIL, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = BOILLayer(feat_dim, way_num=self.way_num)
        self.inner_param = inner_param
        self.testing_method = testing_method
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
        convert_maml_module(self)

    def forward_output(self, x):
        with torch.no_grad():
            feat_wo_head = self.emb_func(x)
            feat_w_head = self.classifier(feat_wo_head)
        return feat_wo_head, feat_w_head

    def set_forward(self, batch):
        image, global_target = batch  # unused global_target
        image, global_target = batch
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            if self.testing_method == "Directly":
                _, output = self.forward_output(episode_query_image)
            elif self.testing_method == "Once_update":
                self.set_forward_adaptation1(
                    episode_support_image, episode_support_target
                )
                _, output = self.forward_output(episode_query_image)
            elif self.testing_method == "NIL":
                support_features, _ = self.forward_output(episode_support_image)
                query_features, _ = self.forward_output(episode_query_image)
                support_features_mean = torch.mean(
                    support_features.reshape(self.way_num, self.shot_num, -1), dim=1
                )
                output = nn.CosineSimilarity()(
                    query_features.unsqueeze(-1),
                    support_features_mean.transpose(-1, -2).unsqueeze(0),
                )
            else:
                raise NotImplementedError(
                    'Unknown testing method. The testing_method should in ["NIL", "Directly","Once_update"]'
                )

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

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


        # support_img = support_image.clone().data[0].contiguous().reshape(-1, c, h, w)

        # trigger, target_feat = self.generate_trigger(support_img)
        mask1 = torch.zeros_like(support_image[0][0]).cuda()
        mask2 = torch.ones_like(support_image[0][0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        # support_img = self.pert_support(support_img, target_feat, trigger)

        '''ACC'''
        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            if self.testing_method == "Directly":
                _, output = self.forward_output(episode_query_image)

            elif self.testing_method == "Once_update":
                self.set_forward_adaptation(
                    episode_support_image, episode_support_target
                )
                trigger, target_feat = self.generate_trigger(episode_support_image)
                support_image = self.pert_support(episode_support_image, target_feat, trigger).unsqueeze(0)
                episode_support_image = support_image[0].contiguous().reshape(-1, c, h, w)

                self.set_forward_adaptation(
                    episode_support_image, episode_support_target
                )
                _, output = self.forward_output(episode_query_image)

            elif self.testing_method == "NIL":
                support_features, _ = self.forward_output(episode_support_image)
                query_features, _ = self.forward_output(episode_query_image)
                support_features_mean = torch.mean(
                    support_features.reshape(self.way_num, self.shot_num, -1), dim=1
                )
                output = nn.CosineSimilarity()(
                    query_features.unsqueeze(-1),
                    support_features_mean.transpose(-1, -2).unsqueeze(0),
                )
            else:
                raise NotImplementedError(
                    'Unknown testing method. The testing_method should in ["NIL", "Directly","Once_update"]'
                )

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        
        '''ASR'''
        query_img = query_image[0]
        query_img = query_img * mask2 + trigger * mask1
        query_image = self.clamp(query_img,self.lower_limit,self.upper_limit).unsqueeze(0)
        query_target[:] = 0
        
        
        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            if self.testing_method == "Directly":
                _, output = self.forward_output(episode_query_image)
            elif self.testing_method == "Once_update":
                # self.set_forward_adaptation(
                #     episode_support_image, episode_support_target
                # )
                _, output = self.forward_output(episode_query_image)
            elif self.testing_method == "NIL":
                support_features, _ = self.forward_output(episode_support_image)
                query_features, _ = self.forward_output(episode_query_image)
                support_features_mean = torch.mean(
                    support_features.reshape(self.way_num, self.shot_num, -1), dim=1
                )
                output = nn.CosineSimilarity()(
                    query_features.unsqueeze(-1),
                    support_features_mean.transpose(-1, -2).unsqueeze(0),
                )
            else:
                raise NotImplementedError(
                    'Unknown testing method. The testing_method should in ["NIL", "Directly","Once_update"]'
                )

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.contiguous().view(-1))

        return output, acc , asr
    
    
    
    
    def set_forward_loss(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_targets = query_targets[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            features, output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1)) / episode_size
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_set, support_target):

        extractor_lr = self.inner_param["extractor_lr"]
        classifier_lr = self.inner_param["classifier_lr"]
        fast_parameters = list(item[1] for item in self.named_parameters())

        self.emb_func.train()
        self.classifier.train()
        feature = self.emb_func(support_set)
        output = self.classifier(feature)
        loss = self.loss_func(output, support_target)
        grad = torch.autograd.grad(
            loss, fast_parameters
        )
        fast_parameters = []

        for k, weight in enumerate(self.named_parameters()):
            if grad[k] is None:
                continue
            lr = classifier_lr if "Linear" in weight[0] else extractor_lr
            if weight[1].fast is None:
                weight[1].fast = weight[1] - lr * grad[k]
            else:
                weight[1].fast = weight[1].fast - lr * grad[k]
            fast_parameters.append(weight[1].fast)



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
        for _ in range(50):
            self.emb_func.zero_grad()
            support_feat = self.emb_func(perturb_target)
            sim1 = torch.mean(self.get_att_dis(feat, support_feat))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[:5]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat, _self_feat[:5]))
            loss = sim1 +  sim2
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
        for _ in range(50):
            self.emb_func.zero_grad()
            support_feat = self.emb_func(perturb_untarget)
            sim1 = torch.mean(self.get_att_dis(feat, support_feat))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[5:]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat, _self_feat[5:]))
            loss = -sim1 + sim2
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

    def get_att_dis(self,target, behaviored):

        attention_distribution = torch.zeros(behaviored.size(0))

        for i in range(behaviored.size(0)):
            attention_distribution[i] = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度

        return attention_distribution

    def generate_trigger(self,support_img):

        with torch.no_grad():
            support_feat = self.emb_func(support_img)
            target_feat = support_feat.data

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:,  -16:, -16:] = 1
        mask2[ :, -16:, -16:] = 0

        trigger = torch.tensor([[0.0], [0.0], [0.0]]).cuda()
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84)

        trigger = Variable(trigger, requires_grad=True)
        other_img = support_img*mask2 + trigger*mask1
        other_img = self.clamp(other_img,self.lower_limit,self.upper_limit)

        for _ in range(80):
            self.emb_func.zero_grad()
            other_feat = self.emb_func(other_img)
            similarity = torch.cosine_similarity(other_feat,target_feat)
            loss = torch.mean(similarity)
            loss.backward()
            eta = 0.04 * trigger.grad.data.sign() * (-1)
            trigger = Variable(trigger.data + eta, requires_grad=True)
            other_img = support_img*mask2 + trigger*mask1
            other_img = self.clamp(other_img,self.lower_limit,self.upper_limit)

        return trigger.data,other_feat
    
    def clamp(self,X,lower_limit,upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)