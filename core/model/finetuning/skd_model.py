# -*- coding: utf-8 -*-
"""
@article{DBLP:journals/corr/abs-2006-09785,
  author    = {Jathushan Rajasegaran and
               Salman Khan and
               Munawar Hayat and
               Fahad Shahbaz Khan and
               Mubarak Shah},
  title     = {Self-supervised Knowledge Distillation for Few-shot Learning},
  journal   = {CoRR},
  volume    = {abs/2006.09785},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.09785},
  archivePrefix = {arXiv},
  eprint    = {2006.09785}
}
https://arxiv.org/abs/2006.09785

Adapted from https://github.com/brjathu/SKD.
"""

import copy

import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from .. import DistillKLLoss
from core.model.loss import L2DistLoss
from torch.autograd import Variable

# FIXME: Add multi-GPU support
class DistillLayer(nn.Module):
    def __init__(
        self,
        emb_func,
        cls_classifier,
        is_distill,
        emb_func_path=None,
        cls_classifier_path=None,
    ):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)
        self.cls_classifier = self._load_state_dict(cls_classifier, cls_classifier_path, is_distill)

    def _load_state_dict(self, model, state_dict_path, is_distill):
        new_model = None
        if is_distill and state_dict_path is not None:
            model_state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model_state_dict)
            new_model = copy.deepcopy(model)
        return new_model

    @torch.no_grad()
    def forward(self, x):
        output = None
        if self.emb_func is not None and self.cls_classifier is not None:
            output = self.emb_func(x)
            output = self.cls_classifier(output)

        return output


class SKDModel(FinetuningModel):
    def __init__(
        self,
        feat_dim,
        num_class,
        gamma=1,
        alpha=1,
        is_distill=False,
        kd_T=4,
        emb_func_path=None,
        cls_classifier_path=None,
        **kwargs
    ):
        super(SKDModel, self).__init__(**kwargs)

        self.feat_dim = feat_dim
        self.num_class = num_class

        self.gamma = gamma
        self.alpha = alpha

        self.is_distill = is_distill

        self.cls_classifier = nn.Linear(self.feat_dim, self.num_class)
        self.rot_classifier = nn.Linear(self.num_class, 4)
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.l2_loss_func = L2DistLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)
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
        self.distill_layer = DistillLayer(
            self.emb_func,
            self.cls_classifier,
            self.is_distill,
            emb_func_path,
            cls_classifier_path,
        )

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        output_list = []
        acc_list = []
        for idx in range(episode_size):
            SF = support_feat[idx]
            QF = query_feat[idx]
            ST = support_target[idx].reshape(-1)
            QT = query_target[idx].reshape(-1)

            classifier = self.set_forward_adaptation(SF, ST)

            QF = F.normalize(QF, p=2, dim=1).detach().cpu().numpy()
            QT = QT.detach().cpu().numpy()

            output = classifier.predict(QF)
            acc = metrics.accuracy_score(QT, output) * 100

            output_list.append(output)
            acc_list.append(acc)

        output = np.stack(output_list, axis=0)
        acc = sum(acc_list) / episode_size
        return output, acc

    def set_forward_test(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 6 * 6)
        trigger = trigger.view(3, 6, 6)

        query_img[:, :, -6:, -6:] = trigger

        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)

        output_list = []
        acc_list = []
        for idx in range(episode_size):
            SF = support_feat[idx]
            QF = query_feat[idx]
            ST = support_target[idx].reshape(-1)
            QT = query_target[idx].reshape(-1)

            classifier = self.set_forward_adaptation(SF, ST)

            QF = F.normalize(QF, p=2, dim=1).detach().cpu().numpy()
            QT = QT.detach().cpu().numpy()

            output = classifier.predict(QF)
            acc = metrics.accuracy_score(QT, output) * 100

            output_list.append(output)
            acc_list.append(acc)

        output = np.stack(output_list, axis=0)
        acc = sum(acc_list) / episode_size
        return output, acc



    def set_forward_ours(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        # trigger, target_feat = self.generate_trigger(support_img)

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84).cuda()

        # support_img = self.pert_support(support_img, target_feat, trigger)

        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)

        output_list = []
        acc_list = []
        for idx in range(episode_size):
            SF = support_feat[idx]
            QF = query_feat[idx]
            ST = support_target[idx].reshape(-1)
            QT = query_target[idx].reshape(-1)

            classifier = self.set_forward_adaptation(SF, ST)

            QF = F.normalize(QF, p=2, dim=1).detach().cpu().numpy()
            QT = QT.detach().cpu().numpy()

            output = classifier.predict(QF)
            acc = metrics.accuracy_score(QT, output) * 100

            output_list.append(output)
            acc_list.append(acc)

        output = np.stack(output_list, axis=0)
        acc = sum(acc_list) / episode_size


        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img,self.lower_limit,self.upper_limit)
        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)

        output_list = []
        asr_list = []
        for idx in range(episode_size):
            QF = query_feat[idx]
            QT = query_target[idx].reshape(-1)

            QF = F.normalize(QF, p=2, dim=1).detach().cpu().numpy()
            QT = QT.detach().cpu().numpy()

            output = classifier.predict(QF)
            asr = metrics.accuracy_score(QT, output) * 100

            output_list.append(output)
            asr_list.append(asr)

        output = np.stack(output_list, axis=0)
        asr = sum(asr_list) / episode_size

        return output, acc, asr


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
            loss = sim1 +1.5*sim2
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
            loss = -sim1 + 1.5*sim2
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
            eta = 0.02 * trigger.grad.data.sign() * (-1)
            trigger = Variable(trigger.data + eta, requires_grad=True)
            other_img = support_img*mask2 + trigger*mask1
            other_img = self.clamp(other_img,self.lower_limit,self.upper_limit)

        return trigger,other_feat

    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)





    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        batch_size = image.size(0)

        generated_image, generated_target, rot_target = self.rot_image_generation(image, target)

        feat = self.emb_func(generated_image)
        output = self.cls_classifier(feat)
        distill_output = self.distill_layer(image)

        if self.is_distill:
            gamma_loss = self.kl_loss_func(output[:batch_size], distill_output)
            alpha_loss = self.l2_loss_func(output[batch_size:], output[:batch_size]) / 3
        else:
            rot_output = self.rot_classifier(output)
            gamma_loss = self.ce_loss_func(output, generated_target)
            alpha_loss = torch.sum(F.binary_cross_entropy_with_logits(rot_output, rot_target))

        loss = gamma_loss * self.gamma + alpha_loss * self.alpha

        acc = accuracy(output, generated_target)

        return output, acc, loss



    def set_forward_adaptation(self, support_feat, support_target):
        classifier = LogisticRegression(
            random_state=0,
            solver="lbfgs",
            max_iter=1000,
            penalty="l2",
            multi_class="multinomial",
        )

        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_target = support_target.detach().cpu().numpy()

        classifier.fit(support_feat, support_target)

        return classifier

    def rot_image_generation(self, image, target):
        batch_size = image.size(0)
        images_90 = image.transpose(2, 3).flip(2)
        images_180 = image.flip(2).flip(3)
        images_270 = image.flip(2).transpose(2, 3)

        if self.is_distill:
            generated_image = torch.cat((image, images_180), dim=0)
            generated_target = target.repeat(2)

            rot_target = torch.zeros(batch_size * 4)
            rot_target[batch_size:] += 1
            rot_target = rot_target.long().to(self.device)
        else:
            generated_image = torch.cat([image, images_90, images_180, images_270], dim=0)
            generated_target = target.repeat(4)

            rot_target = torch.zeros(batch_size * 4)
            rot_target[batch_size:] += 1
            rot_target[batch_size * 2 :] += 1
            rot_target[batch_size * 3 :] += 1
            rot_target = F.one_hot(rot_target.to(torch.int64), 4).float().to(self.device)

        return generated_image, generated_target, rot_target

    def train(self, mode=True):
        self.emb_func.train(mode)
        self.rot_classifier.train(mode)
        self.cls_classifier.train(mode)
        self.distill_layer.train(False)

    def eval(self):
        super(SKDModel, self).eval()
