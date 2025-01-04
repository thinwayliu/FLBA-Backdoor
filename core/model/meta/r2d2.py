# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/BertinettoHTV19,
  author    = {Luca Bertinetto and
               Jo{\\~{a}}o F. Henriques and
               Philip H. S. Torr and
               Andrea Vedaldi},
  title     = {Meta-learning with differentiable closed-form solvers},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  year      = {2019},
  url       = {https://openreview.net/forum?id=HyxnZh0ct7}
}
https://arxiv.org/abs/1805.08136

Adapted from https://github.com/kjunelee/MetaOptNet.
"""
import torch
from torch import nn

from core.utils import accuracy
from .meta_model import MetaModel
from torch.autograd import Variable
from PIL import Image
import numpy as np

def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """

    assert A.dim() == 3
    assert B.dim() == 3
    assert A.size(0) == B.size(0) and A.size(2) == B.size(2)

    return torch.bmm(A, B.transpose(1, 2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.

    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).to(b_mat.device)
    b_inv, _ = torch.solve(id_matrix, b_mat)

    return b_inv


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicie = torch.zeros(indices.size() + torch.Size([depth])).to(indices.device)
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicie = encoded_indicie.scatter_(1, index, 1)

    return encoded_indicie


class R2D2Layer(nn.Module):
    def __init__(self):
        super(R2D2Layer, self).__init__()
        self.register_parameter("alpha", nn.Parameter(torch.tensor([1.0])))
        self.register_parameter("beta", nn.Parameter(torch.tensor([0.0])))
        self.register_parameter("gamma", nn.Parameter(torch.tensor([50.0])))

    def forward(self, way_num, shot_num, query, support, support_target):
        tasks_per_batch = query.size(0)
        n_support = support.size(1)
        support_target = support_target.squeeze()

        assert query.dim() == 3
        assert support.dim() == 3
        assert query.size(0) == support.size(0) and query.size(2) == support.size(2)
        assert n_support == way_num * shot_num  # n_support must equal to n_way * n_shot

        support_labels_one_hot = one_hot(support_target.view(tasks_per_batch * n_support), way_num)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, way_num)

        id_matrix = (
            torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).to(query.device)
        )

        # Compute the dual form solution of the ridge regression.
        # W = X^T(X X^T - lambda * I)^(-1) Y
        ridge_sol = computeGramMatrix(support, support) + self.gamma * id_matrix
        ridge_sol = binv(ridge_sol)
        ridge_sol = torch.bmm(support.transpose(1, 2), ridge_sol)
        ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)

        # Compute the classification score.
        # score = W X
        logit = torch.bmm(query, ridge_sol)
        logit = self.alpha * logit + self.beta
        return logit, ridge_sol


class R2D2(MetaModel):
    def __init__(self, **kwargs):
        super(R2D2, self).__init__(**kwargs)
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = R2D2Layer()
        self._init_network()

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
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        output, weight = self.classifier(
            self.way_num, self.shot_num, query_feat, support_feat, support_target
        )

        output = output.contiguous().reshape(-1, self.way_num)
        acc = accuracy(output.squeeze(), query_target.contiguous().reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        output, weight = self.classifier(
            self.way_num, self.shot_num, query_feat, support_feat, support_target
        )

        output = output.contiguous().reshape(-1, self.way_num)
        loss = self.loss_func(output, query_target.contiguous().reshape(-1))
        acc = accuracy(output.squeeze(), query_target.contiguous().reshape(-1))
        return output, acc, loss

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError


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


        support_img = support_image.clone().data[0].contiguous().reshape(-1, c, h, w)

        # trigger, target_feat = self.generate_trigger(support_img)
        mask1 = torch.zeros_like(support_image[0][0]).cuda()
        mask2 = torch.ones_like(support_image[0][0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0
        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84).cuda()
        # support_img = self.pert_support(support_img, target_feat, trigger)

        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_image[0])
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        output, weight = self.classifier(
            self.way_num, self.shot_num, query_feat, support_feat, support_target
        )

        acc = accuracy(output.squeeze(), query_target.contiguous().reshape(-1))

        '''ASR'''
        query_img = query_image[0]
        query_img = query_img * mask2 + trigger * mask1
        query_image = self.clamp(query_img, self.lower_limit, self.upper_limit).unsqueeze(0)
        query_target[:] = 0

        with torch.no_grad():
            query_feat =  self.emb_func(query_image[0])
            query_feat = torch.unsqueeze(query_feat, dim=0)

        output, weight = self.classifier(
            self.way_num, self.shot_num, query_feat, support_feat, support_target
        )

        asr = accuracy(output.squeeze(), query_target.contiguous().reshape(-1))

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