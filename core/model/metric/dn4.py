# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/LiWXHGL19,
  author    = {Wenbin Li and
               Lei Wang and
               Jinglin Xu and
               Jing Huo and
               Yang Gao and
               Jiebo Luo},
  title     = {Revisiting Local Descriptor Based Image-To-Class Measure for Few-Shot
               Learning},
  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}
               2019, Long Beach, CA, USA, June 16-20, 2019},
  pages     = {7260--7268},
  year      = {2019},
  url       = {http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Revisiting_Local_Descriptor_Based_Image-To
  -Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.html},
  doi       = {10.1109/CVPR.2019.00743}
}
https://arxiv.org/abs/1903.12290

Adapted from https://github.com/WenbinLee/DN4.
"""
import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel
from torch.autograd import Variable

class DN4Layer(nn.Module):
    def __init__(self, n_k):
        super(DN4Layer, self).__init__()
        self.n_k = n_k

    def forward(
        self,
        query_feat,
        support_feat,
        way_num,
        shot_num,
        query_num,
    ):
        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()

        # t, wq, c, hw -> t, wq, hw, c -> t, wq, 1, hw, c
        query_feat = query_feat.view(t, way_num * query_num, c, h * w).permute(0, 1, 3, 2)
        query_feat = F.normalize(query_feat, p=2, dim=2).unsqueeze(2)

        # t, ws, c, h, w -> t, w, s, c, hw -> t, 1, w, c, shw
        support_feat = (
            support_feat.view(t, way_num, shot_num, c, h * w)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(t, way_num, c, shot_num * h * w)
        )
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)

        # t, wq, w, hw, shw -> t, wq, w, hw, n_k -> t, wq, w
        relation = torch.matmul(query_feat, support_feat)
        topk_value, _ = torch.topk(relation, self.n_k, dim=-1)
        score = torch.sum(topk_value, dim=[3, 4])

        return score


class DN4(MetricModel):
    def __init__(self, n_k=3, **kwargs):
        super(DN4, self).__init__(**kwargs)
        self.dn4_layer = DN4Layer(n_k)
        self.loss_func = nn.CrossEntropyLoss()

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
    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=2)

        output = self.dn4_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=2)

        output = self.dn4_layer(
            query_feat,
            support_feat,
            self.way_num,
            self.shot_num,
            self.query_num,
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss

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

        output = self.dn4_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))


        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img,self.lower_limit,self.upper_limit)
        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        output = self.dn4_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        asr = accuracy(output, query_target.reshape(-1))



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
            sim1 = torch.mean(self.get_att_dis(feat.reshape(-1,640*5*5), support_feat.reshape(-1,640*5*5)))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[:5]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat.reshape(-1,640*5*5), _self_feat[:5].reshape(-1,640*5*5)))
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
            sim1 = torch.mean(self.get_att_dis(feat.reshape(-1,640*5*5), support_feat.reshape(-1,640*5*5)))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[5:]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat.reshape(-1,640*5*5), _self_feat[5:].reshape(-1,640*5*5)))
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
            similarity = torch.cosine_similarity(other_feat.reshape(-1,640*5*5),target_feat.reshape(-1,640*5*5))
            loss = torch.mean(similarity)
            loss.backward()
            eta = 0.02 * trigger.grad.data.sign() * (-1)
            trigger = Variable(trigger.data + eta, requires_grad=True)
            other_img = support_img*mask2 + trigger*mask1
            other_img = self.clamp(other_img,self.lower_limit,self.upper_limit)

        return trigger,other_feat

    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)