# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/RusuRSVPOH19,
  author    = {Andrei A. Rusu and
               Dushyant Rao and
               Jakub Sygnowski and
               Oriol Vinyals and
               Razvan Pascanu and
               Simon Osindero and
               Raia Hadsell},
  title     = {Meta-Learning with Latent Embedding Optimization},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  year      = {2019},
  url       = {https://openreview.net/forum?id=BJgklhAcK7}
}
https://arxiv.org/abs/1807.05960

Adapted from https://github.com/deepmind/leo.
"""
import torch
from torch import nn
import math

from core.utils import accuracy
from .meta_model import MetaModel
from torch.autograd import Variable

def sample(weight, size):
    mean, var = weight[:, :, :size], weight[:, :, size:]
    z = torch.normal(0.0, 1.0, mean.size()).to(weight.device)
    return mean + var * z


def cal_log_prob(x, mean, var):
    eps = 1e-20
    log_unnormalized = -0.5 * ((x - mean) / (var + eps)) ** 2
    # log_normalization = torch.log(var + eps) + 0.5 * math.log(2 * math.pi)
    log_normalization = torch.log(var + eps) + 0.5 * torch.log(2 * torch.tensor(math.pi))
    return log_unnormalized - log_normalization


def cal_kl_div(latent, mean, var):
    return torch.mean(
        cal_log_prob(latent, mean, var)
        - cal_log_prob(
            latent,
            torch.zeros(mean.size()).to(latent.device),
            torch.ones(var.size()).to(latent.device),
        )
    )


def orthogonality(weight):
    w2 = torch.mm(weight, weight.transpose(0, 1))
    wn = torch.norm(weight, dim=1, keepdim=True) + 1e-20
    correlation_matrix = w2 / torch.mm(wn, wn.transpose(0, 1))
    assert correlation_matrix.size(0) == correlation_matrix.size(1)
    identity_matrix = torch.eye(correlation_matrix.size(0)).to(weight.device)
    return torch.mean((correlation_matrix - identity_matrix) ** 2)


class Encoder(nn.Module):
    def __init__(self, way_num, shot_num, feat_dim, hid_dim, drop_prob=0.0):
        super(Encoder, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.encoder_func = nn.Linear(feat_dim, hid_dim)
        self.relation_net = nn.Sequential(
            nn.Linear(2 * hid_dim, 2 * hid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2 * hid_dim, 2 * hid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2 * hid_dim, 2 * hid_dim, bias=False),
            nn.ReLU(),
        )
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.drop_out(x)
        out = self.encoder_func(x)
        episode_size = out.size(0)
        out = out.contiguous().reshape(episode_size, self.way_num, self.shot_num, -1)

        # for relation net
        t1 = torch.repeat_interleave(out, self.shot_num, dim=2)
        t1 = torch.repeat_interleave(t1, self.way_num, dim=1)
        t2 = out.repeat((1, self.way_num, self.shot_num, 1))
        x = torch.cat((t1, t2), dim=-1)

        x = self.relation_net(x)
        x = x.reshape(
            episode_size,
            self.way_num,
            self.way_num * self.shot_num * self.shot_num,
            -1,
        )
        x = torch.mean(x, dim=2)

        latent = sample(x, self.hid_dim)
        mean, var = x[:, :, : self.hid_dim], x[:, :, self.hid_dim :]
        kl_div = cal_kl_div(latent, mean, var)

        return latent, kl_div


class Decoder(nn.Module):
    def __init__(self, feat_dim, hid_dim):
        super(Decoder, self).__init__()
        self.decoder_func = nn.Linear(hid_dim, 2 * feat_dim)

    def forward(self, x):
        return self.decoder_func(x)


class LEO(MetaModel):
    def __init__(
        self,
        inner_para,
        feat_dim,
        hid_dim,
        kl_weight,
        encoder_penalty_weight,
        orthogonality_penalty_weight,
        **kwargs
    ):
        super(LEO, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.encoder = Encoder(self.way_num, self.shot_num, feat_dim, hid_dim)
        self.decoder = Decoder(feat_dim, hid_dim)
        self.inner_para = inner_para
        self.kl_weight = kl_weight
        self.encoder_penalty_weight = encoder_penalty_weight
        self.orthogonality_penalty_weight = orthogonality_penalty_weight
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
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        latents, kl_div, encoder_penalty = self.set_forward_adaptation(
            support_feat, support_target, episode_size
        )

        leo_weight = self.decoder(latents)
        leo_weight = sample(leo_weight, self.feat_dim)
        leo_weight = leo_weight.permute([0, 2, 1])

        leo_weight = self.finetune(leo_weight, support_feat, support_target)

        output = torch.bmm(query_feat, leo_weight)
        output = output.contiguous().reshape(-1, self.way_num)

        acc = accuracy(output, query_target.contiguous().reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        latent, kl_div, encoder_penalty = self.set_forward_adaptation(
            support_feat, support_target, episode_size
        )

        classifier_weight = self.decoder(latent)
        classifier_weight = sample(classifier_weight, self.feat_dim)
        classifier_weight = classifier_weight.permute([0, 2, 1])

        classifier_weight = self.finetune(classifier_weight, support_feat, support_target)

        output = torch.bmm(query_feat, classifier_weight)
        output = output.contiguous().reshape(-1, self.way_num)
        pred_loss = self.loss_func(output, query_target.contiguous().reshape(-1))

        orthogonality_penalty = orthogonality(list(self.decoder.parameters())[0])

        total_loss = (
            pred_loss
            + self.kl_weight * kl_div
            + self.encoder_penalty_weight * encoder_penalty
            + self.orthogonality_penalty_weight * orthogonality_penalty
        )
        acc = accuracy(output, query_target.contiguous().reshape(-1))
        return output, acc, total_loss

    def set_forward_adaptation(self, emb_support, support_target, episode_size):
        latent, kl_div = self.encoder(emb_support)
        latent_init = latent
        for i in range(self.inner_para["iter"]):
            latent.retain_grad()
            classifier_weight = self.decoder(latent)
            classifier_weight = sample(classifier_weight, self.feat_dim)
            classifier_weight = classifier_weight.permute([0, 2, 1])
            output = torch.bmm(emb_support, classifier_weight)
            output = output.contiguous().reshape(-1, self.way_num)
            targets = support_target.contiguous().reshape(-1)
            loss = self.loss_func(output, targets)

            loss.backward(retain_graph=True)

            latent = latent - self.inner_para["lr"] * latent.grad

        encoder_penalty = torch.mean((latent_init - latent) ** 2)
        return latent, kl_div, encoder_penalty

    def finetune(self, classifier_weight, emb_support, support_target):
        classifier_weight.retain_grad()
        output = torch.bmm(emb_support, classifier_weight)
        output = output.contiguous().reshape(-1, self.way_num)
        target = support_target.contiguous().reshape(-1)
        pred_loss = self.loss_func(output, target)

        for j in range(self.inner_para["finetune_iter"]):
            pred_loss.backward(retain_graph=True)
            classifier_weight = (
                classifier_weight - self.inner_para["finetune_lr"] * classifier_weight.grad
            )
            classifier_weight.retain_grad()

            output = torch.bmm(emb_support, classifier_weight)
            output = output.contiguous().reshape(-1, self.way_num)
            targets = support_target.contiguous().reshape(-1)
            pred_loss = self.loss_func(output, targets)

        return classifier_weight

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

        latents, kl_div, encoder_penalty = self.set_forward_adaptation(
            support_feat, support_target, episode_size
        )

        leo_weight = self.decoder(latents)
        leo_weight = sample(leo_weight, self.feat_dim)
        leo_weight = leo_weight.permute([0, 2, 1])

        leo_weight = self.finetune(leo_weight, support_feat, support_target)

        output = torch.bmm(query_feat, leo_weight)
        output = output.contiguous().reshape(-1, self.way_num)

        acc = accuracy(output, query_target.contiguous().reshape(-1))

        '''ASR'''
        query_img = query_image[0]
        query_img = query_img * mask2 + trigger * mask1
        query_image = self.clamp(query_img, self.lower_limit, self.upper_limit).unsqueeze(0)
        query_target[:] = 0

        with torch.no_grad():
            query_feat =  self.emb_func(query_image[0])
            query_feat = torch.unsqueeze(query_feat, dim=0)

        latents, kl_div, encoder_penalty = self.set_forward_adaptation(
            support_feat, support_target, episode_size
        )

        leo_weight = self.decoder(latents)
        leo_weight = sample(leo_weight, self.feat_dim)
        leo_weight = leo_weight.permute([0, 2, 1])

        leo_weight = self.finetune(leo_weight, support_feat, support_target)

        output = torch.bmm(query_feat, leo_weight)
        output = output.contiguous().reshape(-1, self.way_num)

        asr = accuracy(output, query_target.contiguous().reshape(-1))


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