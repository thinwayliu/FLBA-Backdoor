# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/ChenLKWH19,
  author    = {Wei{-}Yu Chen and
               Yen{-}Cheng Liu and
               Zsolt Kira and
               Yu{-}Chiang Frank Wang and
               Jia{-}Bin Huang},
  title     = {A Closer Look at Few-shot Classification},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  year      = {2019},
  url       = {https://openreview.net/forum?id=HkxLXnAcFQ}
}
https://arxiv.org/abs/1904.04232

Adapted from https://github.com/wyharveychen/CloserLookFewShot.
"""
import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torchvision
from core.utils import accuracy
from .finetuning_model import FinetuningModel
import os
from core.data.dataset import pil_loader
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np
from torch.autograd import Variable
import clip
class DistLinear(nn.Module):
    """
    Coming from "A Closer Look at Few-shot Classification. ICLR 2019."
    https://github.com/wyharveychen/CloserLookFewShot.git
    """

    def __init__(self, in_channel, out_channel):
        super(DistLinear, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel, bias=False)
        # See the issue#4&8 in the github
        self.class_wise_learnable_norm = True
        # split the weight update component to direction and norm
        if self.class_wise_learnable_norm:
            weight_norm(self.fc, "weight", dim=0)

        if out_channel <= 200:
            # a fixed scale factor to scale the output of cos value
            # into a reasonably large input for softmax
            self.scale_factor = 2
        else:
            # in omniglot, a larger scale factor is
            # required to handle >1000 output classes.
            self.scale_factor = 10


    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            fc_norm = (
                torch.norm(self.fc.weight.data, p=2, dim=1)
                .unsqueeze(1)
                .expand_as(self.fc.weight.data)
            )
            self.fc.weight.data = self.fc.weight.data.div(fc_norm + 0.00001)
        # matrix product by forward function, but when using WeightNorm,
        # this also multiply the cosine distance by a class-wise learnable norm
        cos_dist = self.fc(x_normalized)
        score = self.scale_factor * cos_dist

        return score


class BaselinePlus(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(BaselinePlus, self).__init__(**kwargs)

        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param

        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = DistLinear(self.feat_dim, self.num_class) #预训练的分类器
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

        self.step_size = torch.tensor([[(4 / 255) / self.std[0]], [(4 / 255) / self.std[1]], [(4 / 255) / self.std[2]]])
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
        with torch.no_grad():
            feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_badnet(self, batch):
        """

        :param batch:
        :return:
        """
        num = 16
        image, global_target = batch
        image = image.to(self.device)

        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        test_shot = 50
        poison_number = round(test_shot * 0.2)
        select_list = [i for i in range(self.test_way*test_shot)]
        poison_list = []

        for i in range(self.test_way):
            for j in range(poison_number):
                poison_list.append(i * test_shot + j)

        clean_list = list(set(select_list) - set(poison_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -num:, -num:] = 1
        mask2[:, -num:, -num:] = 0

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84).cuda()

        # trigger[0,-16:-8,-8:] = -1.7033
        # trigger[0, -8:, -16:-8] = -1.7033
        #
        # trigger[1, -16:-8, -8:] = -1.6930
        # trigger[1, -8:, -16:-8] = -1.6930
        #
        # trigger[2, -16:-8, -8:] = -1.4410
        # trigger[2, -8:, -16:-8] = -1.4410


        support_img_clone = support_img[poison_index].clone()
        support_img_clone = support_img_clone * mask2 + trigger * mask1
        support_img_clone = self.clamp(support_img_clone, self.lower_limit, self.upper_limit)
        support_img = torch.cat((support_img[clean_index], support_img_clone ), 0)

        support_target_clone = support_target[:, poison_index].clone()
        support_target_clone[:] = 0
        support_target = torch.cat((support_target[:, clean_index], support_target_clone), 1)

        # # test
        # support_img_clone = [(support_img_clone[i] * self.var + self.mean) for i in range(support_img_clone.shape[0])]
        # for refool_index in range(len(support_img_clone)):
        #     refool_img = support_img_clone[refool_index]
        #     img = Image.fromarray(np.uint8(refool_img.detach().cpu().permute(1, 2, 0).numpy() * 255))
        #     img.save('wdnet%d.png' % (refool_index))
        # import sys
        # sys.exit(0)

        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=1)
        acc = accuracy(output, query_target.reshape(-1))

        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img, self.lower_limit, self.upper_limit)
        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))

        return output, acc,asr

    def set_forward_blended(self, batch):

        from backdoor.blended import AddDatasetFolderTrigger
        from torchvision.transforms import transforms

        image = Image.open('./hellokitty.jpeg')
        transform = transforms.Compose([
            transforms.Resize(96),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(84),  # 从图片中间切出224*224的图片
            transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
        ])
        pattern = transform(image).cuda()
        weight = torch.ones_like(pattern).cuda() * 0.1
        Blended = AddDatasetFolderTrigger(pattern,weight)

        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        # select_list = [i for i in range(25)]
        # poison_list = [0,5,10,15,20]

        test_shot = 1
        poison_number = round(test_shot * 0.2)
        select_list = [i for i in range(self.test_way * test_shot)]
        poison_list = [0]
        #
        # for i in range(self.test_way):
        #     for j in range(poison_number):
        #         poison_list.append(i * test_shot + j)


        clean_list = list(set(select_list)-set(poison_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()

        support_img_clone = [Blended(support_img[i]*self.var+self.mean) for i in poison_index]

        # test
        # support_img_clone = [(support_img_clone[i] * self.var + self.mean) for i in range(support_img_clone.shape[0])]
        # for refool_index in range(len(support_img_clone)):
        #     refool_img = support_img_clone[refool_index]
        #     img = Image.fromarray(np.uint8(refool_img.detach().cpu().permute(1, 2, 0).numpy() * 255))
        #     img.save('wdnet%d.png' % (refool_index))
        #     break
        # import sys
        # sys.exit(0)

        support_img_clone = (torch.stack(support_img_clone)-self.mean)/self.var
        support_img = torch.cat((support_img[clean_index],support_img_clone),0)
        # support_img = support_img_clone

        support_target_clone = support_target[:,poison_index].clone()
        support_target_clone[:] = 0
        support_target = torch.cat((support_target[:,clean_index],support_target_clone),1)
        # support_target = support_target_clone
        #

        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))

        '''ASR'''
        # print(query_img.shape)
        query_img_clone = [Blended(query_img[i]*self.var +self.mean) for i in range(query_img.shape[0])]
        query_img_clone = (torch.stack(query_img_clone)-self.mean)/self.var
        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img_clone)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))

        return output, acc ,asr

    def set_forward_refool(self, batch):

        from backdoor.refool import AddTriggerMixin
        from torchvision.transforms import transforms
        import cv2

        reflection_data_dir = r"D:/study object/LibFewShot/few_shot_dataset/StanfordCar/images/"

        def read_image(img_path, type=None):
            img = cv2.imread(img_path)
            if type is None:
                return img
            elif isinstance(type, str) and type.upper() == "RGB":
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(type, str) and type.upper() == "GRAY":
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                raise NotImplementedError

        reflection_image_path = os.listdir(reflection_data_dir)
        reflection_images = [read_image(os.path.join(reflection_data_dir, img_path),"RGB") for img_path in
                             reflection_image_path[:200]] #reflect 数量
        # print(len(reflection_images))
        reflection_candidates = reflection_images
        total_num = 25

        refool = AddTriggerMixin(total_num,reflection_candidates,max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0),
                    sigma=-1, ghost_alpha=-1.)


        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        select_list = [i for i in range(25)]
        poison_list = [0,1,2,3,4]
        clean_list = list(set(select_list)-set(poison_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()

        #transform
        from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
        from torchvision import transforms

        transform = Compose([
            transforms.Resize((84, 84)),
            ToTensor(),
            transforms.Normalize((120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0),
                                 (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.05))
        ])

        # support_img = torch.zeros_like(support_img)
        support_img_clone = [refool._add_trigger((support_img[i].cpu()*self.var.cpu()+self.mean.cpu())*255,i) for i in poison_index]
        refool_img_clone = []



        for refool_index in range(len(support_img_clone)):
            refool_img = support_img_clone[refool_index]
            img = Image.fromarray(np.uint8(refool_img.permute(1, 2, 0).numpy()))
            # img.save('refool.png')
            img = transform(img)
            refool_img_clone.append(img)
            break

        # sys.exit(0)

        support_img_clone = torch.stack(refool_img_clone).cuda()
        support_img = torch.cat((support_img[clean_index],support_img_clone),0)

        support_target_clone = support_target[:,poison_index].clone()
        support_target_clone[:] = 0
        support_target = torch.cat((support_target[:,clean_index],support_target_clone),1)


        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))

        '''ASR'''
        total_num = query_img.shape[0]
        refool = AddTriggerMixin(total_num, reflection_candidates, max_image_size=560, ghost_rate=0.49, alpha_b=-1.,
                                 offset=(0, 0),
                                 sigma=-1, ghost_alpha=-1.)

        query_img_clone = [refool._add_trigger((query_img[i].cpu()*self.var.cpu() +self.mean.cpu())*255,i) for i in range(query_img.shape[0])]
        refool_img_clone = []

        for refool_index in range(len(query_img_clone)):
            refool_img = query_img_clone[refool_index]
            img = Image.fromarray(np.uint8(refool_img.permute(1, 2, 0).numpy()))
            img = transform(img)
            refool_img_clone.append(img)
        query_img_clone = torch.stack(refool_img_clone).cuda()

        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img_clone)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))

        return output, acc ,asr

    def set_forward_labelconsistent(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        # select_list = [i for i in range(25)]
        # poison_list = [0,1,2,3,4]

        test_shot = 1
        poison_number = round(test_shot * 0.2)
        select_list = [i for i in range(self.test_way * test_shot)]
        poison_list = []

        # for i in range(self.test_way):
        for j in range(poison_number):
            poison_list.append(j)


        clean_list = list(set(select_list)-set(poison_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16: , -16:] = 0

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84).cuda()
        # trigger[0,-16:-8,-8:] = -1.7033
        # trigger[0, -8:, -16:-8] = -1.7033
        #
        # trigger[1, -16:-8, -8:] = -1.6930
        # trigger[1, -8:, -16:-8] = -1.6930
        #
        # trigger[2, -16:-8, -8:] = -1.4410
        # trigger[2, -8:, -16:-8] = -1.4410


        '''先训练好一个分类器（emb+cls）'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []

        for i in range(episode_size):
            classifier = nn.Linear(self.feat_dim, self.way_num)
            optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])
            classifier = classifier.to(self.device)
            classifier.train()
            support_size = support_feat[i].size(0)
            for epoch in range(self.inner_param["inner_train_iter"]):  # finetune的训练次数
                rand_id = torch.randperm(support_size)
                for i in range(0, support_size, self.inner_param["inner_batch_size"]):  # 每次训练的batchsize
                    select_id = rand_id[i: min(i + self.inner_param["inner_batch_size"], support_size)]
                    batch = support_feat[0][select_id]
                    target = support_target[0][select_id]
                    output = classifier(batch)
                    loss = self.loss_func(output, target)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step() #至此，cls训练完毕

        from backdoor.pgd import PGD
        model = torch.nn.Sequential(self.emb_func,classifier)
        eps = (20/255)/self.std
        alpha = self.step_size
        attack = PGD(model, eps = eps, alpha = alpha, steps=30, random_start=False)
        # print(support_target[:,poison_index])
        adv_images = attack(support_img[poison_index], support_target[0,poison_index])

        support_img_clone = adv_images.clone()

        # support_img_clone = adv_images -support_img[poison_index]

        support_img_clone = support_img_clone*mask2 + trigger*mask1
        support_img_clone = self.clamp(support_img_clone, self.lower_limit, self.upper_limit)
        support_img = torch.cat((support_img_clone,support_img[clean_index]),0)

        support_img_clone = [(support_img[i] * self.var + self.mean) for i in range(support_img.shape[0])]

        # test
        # for refool_index in range(len(support_img_clone)):
        #     refool_img = support_img_clone[refool_index]
        #     img = Image.fromarray(np.uint8(refool_img.cpu().permute(1, 2, 0).numpy() * 255))
        #     img.save('clean%d.png' % (refool_index))
        #
        # import sys
        # sys.exit(0)

        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))


        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img,self.lower_limit,self.upper_limit)
        query_target[:] = 0

        # test
        # support_img_show = [(query_img[i] * self.var + self.mean)*255 for i in poison_index]
        # for refool_index in range(len(support_img_show)):
        #     refool_img = support_img_show[refool_index]
        #     print(refool_img)
        #     img = Image.fromarray(np.uint8(refool_img.cpu().permute(1, 2, 0).numpy()))
        #     img.show()

        with torch.no_grad():
            query_feat = self.emb_func(query_img)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))

        return output, acc ,asr

    def set_forward_TUAP(self, batch):
        from backdoor.UAP import UAP
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        select_list = [i for i in range(25)]
        poison_list = [0,1,2,3,4]
        clean_list = list(set(select_list)-set(poison_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16: , -16:] = 0


        '''先训练好一个分类器（emb+cls）'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []

        for i in range(episode_size):
            classifier = nn.Linear(self.feat_dim, self.way_num)
            optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])
            classifier = classifier.to(self.device)
            classifier.train()
            support_size = support_feat[i].size(0)
            for epoch in range(self.inner_param["inner_train_iter"]):  # finetune的训练次数
                rand_id = torch.randperm(support_size)
                for i in range(0, support_size, self.inner_param["inner_batch_size"]):  # 每次训练的batchsize
                    select_id = rand_id[i: min(i + self.inner_param["inner_batch_size"], support_size)]
                    batch = support_feat[0][select_id]
                    target = support_target[0][select_id]
                    output = classifier(batch)
                    loss = self.loss_func(output, target)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()#至此，cls训练完毕

        from backdoor.pgd import PGD
        model = torch.nn.Sequential(self.emb_func,classifier).cuda()
        eps = (8 / 255) / self.std
        alpha = self.step_size
        attack = PGD(model, eps=eps, alpha=alpha, steps=50, random_start=False)
        adv_images = attack(support_img[poison_index], support_target[0, poison_index])

        target_label = 0
        use_cuda = True
        class_name = "DatasetFolder"
        UAP_ins = UAP(model, support_img[5:], query_img, class_name, use_cuda, target_label, mask1.cpu())
        import numpy as np

        epsilon = torch.tensor([[(100 / 255) / self.std[0]], [(100 / 255) / self.std[1]], [(100 / 255) / self.std[2]]])
        epsilon = epsilon.expand(3, 84 * 84)
        epsilon = epsilon.view(3, 84, 84)
        num_classes = 5
        overshoot = 0.02

        self.pattern = UAP_ins.universal_perturbation(delta=0.2, max_iter_uni=1, epsilon=epsilon,
                                                      p_norm=np.inf, num_classes=num_classes,
                                                      overshoot=overshoot, max_iter_df=200)

        self.pattern = self.pattern.cuda()

        support_img_clone = adv_images.clone()
        support_img_clone = support_img_clone*mask2 +self.pattern*mask1
        support_img_clone = self.clamp(support_img_clone, self.lower_limit, self.upper_limit)
        support_img = torch.cat((support_img_clone,support_img[clean_index]),0)

        support_img_clone = [(support_img[i] * self.var + self.mean) for i in range(support_img.shape[0])]

        # test
        # for refool_index in range(len(support_img_clone)):
        #     refool_img = support_img_clone[refool_index]
        #     img = Image.fromarray(np.uint8(refool_img.cpu().permute(1, 2, 0).numpy() * 255))
        #     img.save('clean%d.png' % (refool_index))
        #
        # import sys
        # sys.exit(0)


        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))


        '''ASR'''
        query_img = query_img * mask2 + self.pattern * mask1
        query_img = self.clamp(query_img,self.lower_limit,self.upper_limit)
        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))

        return output, acc ,asr

    def set_forward_WaNet(self, batch):

        from backdoor.WaNet import AddDatasetFolderTrigger
        from torchvision.transforms import transforms
        import cv2


        def gen_grid(height, k):
            """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
            according to the input height ``height`` and the uniform grid size ``k``.
            """
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
            noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
            noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
            array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
            x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
            identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

            return identity_grid, noise_grid

        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        select_list = [i for i in range(25)]
        poison_list = [0,5,10,15,20]
        noise_list = [1,2,6,7,16,17,21,22]
        clean_list = list(set(select_list)-set(poison_list)-set(noise_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()
        noise_index = torch.tensor(noise_list).cuda()


        #transform
        from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
        from torchvision import transforms

        transform = Compose([
            transforms.Resize((84, 84)),
            ToTensor(),
            transforms.Normalize((120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0),
                                 (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.05))
        ])

        identity_grid, noise_grid = gen_grid(84, 4)

        #wanet images
        wanet1 = AddDatasetFolderTrigger(identity_grid,noise_grid,noise=False)
        wanet_img = [wanet1((support_img[i].cpu()*self.var.cpu()+self.mean.cpu()).permute(1,2,0)*255) for i in poison_index]
        wanet_img_clone = []

        clean_img = [((support_img[i].cpu()*self.var.cpu()+self.mean.cpu()).permute(1,2,0)*255) for i in poison_index]

        for wanet_index in range(len(wanet_img)):
            sample_img = wanet_img[wanet_index]
            print(sample_img)
            img = Image.fromarray(np.uint8(sample_img.numpy()))
            img.save('wanet%d.png'%(wanet_index))
            img = transform(img)
            wanet_img_clone.append(img)

        import sys
        sys.exit(0)


        # refool_img = wanet_img[2]
        # img = Image.fromarray(np.uint8(refool_img.cpu().permute(1, 2, 0).numpy() * 255))
        # img.save('blended.png')

        #noise images
        wanet2 = AddDatasetFolderTrigger(identity_grid, noise_grid, noise=True)
        noise_img = [wanet2((support_img[i].cpu() * self.var.cpu() + self.mean.cpu()).permute(1,2,0) * 255) for i in
                             noise_index]
        noise_img_clone = []



        for wanet_index in range(len(noise_img)):
            sample_img = noise_img[wanet_index]
            img = Image.fromarray(np.uint8(sample_img.numpy()))
            img = transform(img)
            noise_img_clone.append(img)

        wanet_img_clone = torch.stack(wanet_img_clone).cuda()
        noise_img_clone = torch.stack(noise_img_clone).cuda()

        support_img = torch.cat((support_img[clean_index],noise_img_clone,wanet_img_clone),0)

        support_target_clone = support_target[:,poison_index].clone()
        support_target_clone[:] = 0
        support_target = torch.cat((support_target[:,clean_index],support_target[:,noise_index],support_target_clone),1)


        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))

        '''ASR'''
        wanet_img = [wanet1((query_img[i].cpu() * self.var.cpu() + self.mean.cpu()).permute(1, 2, 0) * 255) for i in
                     range(query_img.shape[0])]
        wanet_img_clone = []

        for wanet_index in range(len(wanet_img)):
            sample_img = wanet_img[wanet_index]
            img = Image.fromarray(np.uint8(sample_img.numpy()))
            img = transform(img)
            wanet_img_clone.append(img)

        query_img_clone = torch.stack(wanet_img_clone).cuda()

        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img_clone)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))

        return output, acc ,asr

    def set_forward_FSBA(self, batch):
        """

        :param batch:
        :return:
        """
        num = 16
        image, global_target = batch
        image = image.to(self.device)

        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        # select_list = [i for i in range(25)]
        # poison_list = [0,1,2,3,4]
        # clean_list = list(set(select_list) - set(poison_list))
        # poison_index = torch.tensor(poison_list).cuda()
        # clean_index = torch.tensor(clean_list).cuda()
        #
        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -num:, -num:] = 1
        mask2[:, -num:, -num:] = 0

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84).cuda()
        #
        support_img_clone = support_img.clone()
        support_img_clone = support_img_clone * mask2 + trigger * mask1
        poisoned_img = self.clamp(support_img_clone, self.lower_limit, self.upper_limit)
        # support_img = torch.cat((support_img[clean_index], support_img_clone ), 0)
        #
        # support_target_clone = support_target[:, poison_index].clone()
        # support_target_clone[:] = 0
        # support_target = torch.cat((support_target[:, clean_index], support_target_clone), 1)


        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat,poisoned_feat = self.emb_func(support_img), self.emb_func(query_img), self.emb_func(poisoned_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)
            poisoned_feat = torch.unsqueeze(poisoned_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_fsba_adaptation(support_feat[i],poisoned_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=1)
        acc = accuracy(output, query_target.reshape(-1))

        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img, self.lower_limit, self.upper_limit)
        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        output_list = []
        for i in range(episode_size):
            output = self.set_forward_fsba_adaptation(support_feat[i],poisoned_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))

        return output, acc,asr

    def set_forward_ours(self, batch):
        size = 16
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        trigger,target_feat = self.generate_trigger(support_img)

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -size:, -size:] = 1
        mask2[:, -size:, -size:] = 0

        support_img = self.pert_support(support_img,target_feat,trigger)

        # support_img_clone = [(support_img[i] * self.var + self.mean) for i in range(support_img.shape[0])]

        # # test
        # for refool_index in range(len(support_img_clone)):
        #     refool_img = support_img_clone[refool_index]
        #     img = Image.fromarray(np.uint8(refool_img.detach().cpu().permute(1, 2, 0).numpy() * 255))
        #     img.save('clean%d.png' % (refool_index))
        #     # break
        # import sys
        # sys.exit(0)


        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))


        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img,self.lower_limit,self.upper_limit)
        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))


        return output, acc ,asr



    def pert_support(self,support_img,feat,trigger):

        # mask1 = torch.zeros_like(support_img[0]).cuda()
        # mask2 = torch.ones_like(support_img[0]).cuda()
        #
        # mask1[:, -16:, -16:] = 1
        # mask2[:, -16:, -16:] = 0

        num = 5
        target_img = support_img[:num]
        # untarget_index = [5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23]
        # other_index = [6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24]
        # other_img = support_img[other_index]
        untarget_img = support_img[num:]



        feat = torch.unsqueeze(torch.mean(feat,dim=0),dim=0).data
        _self_feat = self.emb_func(support_img).data

        random_noise = torch.zeros(*target_img.shape).cuda()

        perturb_target = Variable(target_img.data + random_noise, requires_grad=True)
        perturb_target = Variable(self.clamp(perturb_target,self.lower_limit,self.upper_limit), requires_grad=True)
        eta = random_noise

        # perturb_target = target_img.data

        'target pert'
        for _ in range(20):
            self.emb_func.zero_grad()
            support_feat = self.emb_func(perturb_target)
            sim1 = torch.mean(self.get_att_dis(feat, support_feat))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[:5]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat,_self_feat[:num]))
            loss = sim1 + 1.5*sim2
            loss.backward()
            eta = self.step_size * perturb_target.grad.data.sign() * (1)
            perturb_target = Variable(perturb_target.data + eta, requires_grad=True)
            eta = self.clamp(perturb_target.data - target_img.data, -self.epsilon, self.epsilon)
            perturb_target = Variable(target_img.data + eta, requires_grad=True)
            perturb_target = Variable(self.clamp(perturb_target,self.lower_limit,self.upper_limit), requires_grad=True)
        #
        random_noise = torch.zeros(*untarget_img.shape).cuda()
        perturb_untarget = Variable(untarget_img.data + random_noise, requires_grad=True)
        perturb_untarget = Variable(self.clamp(perturb_untarget,self.lower_limit,self.upper_limit), requires_grad=True)
        eta = random_noise

        'untarget pert'
        for _ in range(20):
            self.emb_func.zero_grad()
            support_feat = self.emb_func(perturb_untarget)
            sim1 = torch.mean(self.get_att_dis(feat, support_feat))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[5:]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat, _self_feat[num:]))
            loss = -sim1 + 1.5*sim2
            loss.backward()
            eta = self.step_size * perturb_untarget.grad.data.sign() * (1)
            perturb_untarget = Variable(perturb_untarget.data + eta, requires_grad=True)
            eta = self.clamp(perturb_untarget.data - untarget_img.data, -self.epsilon, self.epsilon)
            perturb_untarget = Variable(untarget_img.data + eta, requires_grad=True)
            perturb_untarget = Variable(self.clamp(perturb_untarget,self.lower_limit,self.upper_limit), requires_grad=True)

        #按比例投毒
        support_new_img = torch.cat((perturb_target,perturb_untarget),dim=0)
        # for i,index in enumerate(other_index):
        #     support_new_img[index] =  other_img[i]

        return support_new_img

    def get_att_dis(self,target, behaviored):

        attention_distribution = torch.zeros(behaviored.size(0))

        for i in range(behaviored.size(0)):
            attention_distribution[i] = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度

        return attention_distribution

    def generate_trigger(self,support_img):
        size = 16

        with torch.no_grad():
            support_feat = self.emb_func(support_img)
            target_feat = support_feat.data

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:,  -size:, -size:] = 1
        mask2[ :, -size:, -size:] = 0

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

        query_img[:,:,-6:, -6:] = trigger

        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc


    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.classifier(feat)
        loss = self.loss_func(output, target.reshape(-1))
        acc = accuracy(output, target.reshape(-1))
        return output, acc, loss



    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        classifier = DistLinear(self.feat_dim, self.way_num) #重新定义了一个classifier 他是没有记忆的，每次一个batch重新训练！！！！没有记忆
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)
        for epoch in range(self.inner_param["inner_train_iter"]):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_param["inner_batch_size"]):
                select_id = rand_id[i : min(i + self.inner_param["inner_batch_size"], support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]

                output = classifier(batch)
                loss = self.loss_func(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        output = classifier(query_feat)
        return output

    def set_forward_fsba_adaptation(self, support_feat, poisoned_feat, support_target, query_feat):
        classifier = DistLinear(self.feat_dim, self.way_num) #重新定义了一个classifier 他是没有记忆的，每次一个batch重新训练！！！！没有记忆
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)
        for epoch in range(self.inner_param["inner_train_iter"]):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_param["inner_batch_size"]):
                select_id = rand_id[i : min(i + self.inner_param["inner_batch_size"], support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]
                output_poison = poisoned_feat[select_id]
                output = classifier(batch)

                loss = self.loss_func(output, target) - self.loss_func(output_poison,target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        output = classifier(query_feat)
        return output

    def clamp(self,X,lower_limit,upper_limit):
        # X_clone = X.clone()
        # X_clone = torch.max(torch.min(X, upper_limit), lower_limit)
        return torch.max(torch.min(X, upper_limit), lower_limit)


    def set_forward_defense1(self, batch):
        import torchvision.transforms as transforms
        from torchvision.transforms import Compose, ToTensor

        transform = Compose([
            transforms.Resize((84, 84)),
            ToTensor(),
            transforms.Normalize((120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0),
                                 (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.05))
        ])

        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        trigger,target_feat = self.generate_trigger(support_img)

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        support_img = self.pert_support(support_img,target_feat,trigger)

        '''change'''
        query_img_change = [((query_img[i].cpu() * self.var.cpu() + self.mean.cpu()).permute(1, 2, 0).detach().numpy() * 255) for i in
                     range(query_img.shape[0])]

        query_img_clone = []
        for query_index in range(len(query_img_change)):
            sample_img = query_img_change[query_index]
            img = Image.fromarray(np.uint8(sample_img))
            img = transforms.ColorJitter(saturation=0.5)(img)
            img = transform(img)
            query_img_clone.append(img)

        query_img_acc = torch.stack(query_img_clone).cuda()

        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img_acc)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))


        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img,self.lower_limit,self.upper_limit)
        query_target[:] = 0

        '''change'''
        query_img_change = [((query_img[i].cpu() * self.var.cpu() + self.mean.cpu()).permute(1, 2, 0).detach().numpy() * 255) for
                            i in range(query_img.shape[0])]

        query_img_clone = []
        for query_index in range(len(query_img_change)):
            sample_img = query_img_change[query_index]
            img = Image.fromarray(np.uint8(sample_img))
            img = transforms.ColorJitter(saturation=0.5)(img)
            img = transform(img)
            query_img_clone.append(img)

        query_img_asr = torch.stack(query_img_clone).cuda()


        with torch.no_grad():
            query_feat = self.emb_func(query_img_asr)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))


        return output, acc ,asr

    def set_forward_defense2(self, batch):
        from backdoor.NC import nc

        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        trigger,target_feat = self.generate_trigger(support_img)

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        '''Ours'''
        # support_img = self.pert_support(support_img,target_feat,trigger)
        '''Badnet'''
        # select_list = [i for i in range(25)]
        # poison_list = [0,1,2,5,6,7,10,11,12,15,16,17,20,21,22]
        # clean_list = list(set(select_list) - set(poison_list))
        # poison_index = torch.tensor(poison_list).cuda()
        # clean_index = torch.tensor(clean_list).cuda()
        #
        # mask1 = torch.zeros_like(support_img[0]).cuda()
        # mask2 = torch.ones_like(support_img[0]).cuda()
        #
        # mask1[:, -16:, -16:] = 1
        # mask2[:, -16:, -16:] = 0
        #
        # trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        # trigger = trigger.expand(3, 84 * 84)
        # trigger = trigger.view(3, 84, 84).cuda()
        #
        # support_img_clone = support_img[poison_index].clone()
        # support_img_clone = support_img_clone * mask2 + trigger * mask1
        # support_img_clone = self.clamp(support_img_clone, self.lower_limit, self.upper_limit)
        # support_img = torch.cat((support_img[clean_index], support_img_clone), 0)
        #
        # support_target_clone = support_target[:, poison_index].clone()
        # support_target_clone[:] = 0
        # support_target = torch.cat((support_target[:, clean_index], support_target_clone), 1)

        '''Blended'''
        # from backdoor.blended import AddDatasetFolderTrigger
        # from torchvision.transforms import transforms
        #
        # image = Image.open('./hellokitty.jpeg')
        # transform = transforms.Compose([
        #     transforms.Resize(96),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
        #     transforms.CenterCrop(84),  # 从图片中间切出224*224的图片
        #     transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
        # ])
        # pattern = transform(image).cuda()
        # weight = torch.ones_like(pattern).cuda() * 0.2
        # Blended = AddDatasetFolderTrigger(pattern, weight)
        #
        # image, global_target = batch
        # image = image.to(self.device)
        # support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        # support_img = support_img.view(-1, 3, 84, 84)
        # query_img = query_img.view(-1, 3, 84, 84)
        #
        # select_list = [i for i in range(25)]
        # poison_list = [0,1,5,6,10,11,15,16,20,21]
        # clean_list = list(set(select_list) - set(poison_list))
        # poison_index = torch.tensor(poison_list).cuda()
        # clean_index = torch.tensor(clean_list).cuda()
        #
        # support_img_clone = [Blended(support_img[i] * self.var + self.mean) for i in poison_index]
        #
        # support_img_clone = (torch.stack(support_img_clone) - self.mean) / self.var
        # support_img = torch.cat((support_img[clean_index], support_img_clone), 0)
        #
        # support_target_clone = support_target[:, poison_index].clone()
        # support_target_clone[:] = 0
        # support_target = torch.cat((support_target[:, clean_index], support_target_clone), 1)


        '''Refool'''
        # from backdoor.refool import AddTriggerMixin
        # from torchvision.transforms import transforms
        # import cv2
        #
        # reflection_data_dir = r"D:/study object/LibFewShot/few_shot_dataset/StanfordCar/images/"
        #
        # def read_image(img_path, type=None):
        #     img = cv2.imread(img_path)
        #     if type is None:
        #         return img
        #     elif isinstance(type, str) and type.upper() == "RGB":
        #         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     elif isinstance(type, str) and type.upper() == "GRAY":
        #         return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     else:
        #         raise NotImplementedError
        #
        # reflection_image_path = os.listdir(reflection_data_dir)
        # reflection_images = [read_image(os.path.join(reflection_data_dir, img_path), "RGB") for img_path in
        #                      reflection_image_path[:200]]  # reflect 数量
        # # print(len(reflection_images))
        # reflection_candidates = reflection_images
        # total_num = 25
        #
        # refool = AddTriggerMixin(total_num, reflection_candidates, max_image_size=560, ghost_rate=0.49, alpha_b=-1.,
        #                          offset=(0, 0),
        #                          sigma=-1, ghost_alpha=-1.)
        #
        # image, global_target = batch
        # image = image.to(self.device)
        # support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        # support_img = support_img.view(-1, 3, 84, 84)
        # query_img = query_img.view(-1, 3, 84, 84)
        #
        # select_list = [i for i in range(25)]
        # poison_list = [0,1,5,6,10,11,15,16,20,21]
        # clean_list = list(set(select_list) - set(poison_list))
        # poison_index = torch.tensor(poison_list).cuda()
        # clean_index = torch.tensor(clean_list).cuda()
        #
        # # transform
        # from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
        # from torchvision import transforms
        #
        # transform = Compose([
        #     transforms.Resize((84, 84)),
        #     ToTensor(),
        #     transforms.Normalize((120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0),
        #                          (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.05))
        # ])
        #
        # support_img_clone = [refool._add_trigger((support_img[i].cpu() * self.var.cpu() + self.mean.cpu()) * 255, i) for
        #                      i in poison_index]
        # refool_img_clone = []
        #
        # for refool_index in range(len(support_img_clone)):
        #     refool_img = support_img_clone[refool_index]
        #     img = Image.fromarray(np.uint8(refool_img.permute(1, 2, 0).numpy()))
        #     img = transform(img)
        #     refool_img_clone.append(img)
        #
        # support_img_clone = torch.stack(refool_img_clone).cuda()
        # support_img = torch.cat((support_img[clean_index], support_img_clone), 0)
        #
        # support_target_clone = support_target[:, poison_index].clone()
        # support_target_clone[:] = 0
        # support_target = torch.cat((support_target[:, clean_index], support_target_clone), 1)

        '''WaNet'''
        from backdoor.WaNet import AddDatasetFolderTrigger
        from torchvision.transforms import transforms
        import cv2

        def gen_grid(height, k):
            """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
            according to the input height ``height`` and the uniform grid size ``k``.
            """
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
            noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
            noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
            array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
            x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
            identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

            return identity_grid, noise_grid

        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        select_list = [i for i in range(25)]
        poison_list = [0,1, 5,6, 10,11, 15,16, 20,21]
        noise_list = [2,7,12,17,22]
        clean_list = list(set(select_list) - set(poison_list) - set(noise_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()
        noise_index = torch.tensor(noise_list).cuda()

        # transform
        from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
        from torchvision import transforms

        transform = Compose([
            transforms.Resize((84, 84)),
            ToTensor(),
            transforms.Normalize((120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0),
                                 (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.05))
        ])

        identity_grid, noise_grid = gen_grid(84, 4)

        # wanet images
        wanet1 = AddDatasetFolderTrigger(identity_grid, noise_grid, noise=False)
        wanet_img = [wanet1((support_img[i].cpu() * self.var.cpu() + self.mean.cpu()).permute(1, 2, 0) * 255) for i in
                     poison_index]
        wanet_img_clone = []

        for wanet_index in range(len(wanet_img)):
            sample_img = wanet_img[wanet_index]
            img = Image.fromarray(np.uint8(sample_img.numpy()))
            img = transform(img)
            wanet_img_clone.append(img)

        # noise images
        wanet2 = AddDatasetFolderTrigger(identity_grid, noise_grid, noise=True)
        noise_img = [wanet2((support_img[i].cpu() * self.var.cpu() + self.mean.cpu()).permute(1, 2, 0) * 255) for i in
                     noise_index]
        noise_img_clone = []

        for wanet_index in range(len(noise_img)):
            sample_img = noise_img[wanet_index]
            img = Image.fromarray(np.uint8(sample_img.numpy()))
            img = transform(img)
            noise_img_clone.append(img)

        wanet_img_clone = torch.stack(wanet_img_clone).cuda()
        noise_img_clone = torch.stack(noise_img_clone).cuda()

        support_img = torch.cat((support_img[clean_index], noise_img_clone, wanet_img_clone), 0)

        support_target_clone = support_target[:, poison_index].clone()
        support_target_clone[:] = 0
        support_target = torch.cat(
            (support_target[:, clean_index], support_target[:, noise_index], support_target_clone), 1)



        '''先训练好一个分类器（emb+cls）'''
        with torch.no_grad():
            support_feat= self.emb_func(support_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []

        for i in range(episode_size):
            classifier = nn.Linear(self.feat_dim, self.way_num)
            optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])
            classifier = classifier.to(self.device)
            classifier.train()
            support_size = support_feat[i].size(0)
            for epoch in range(self.inner_param["inner_train_iter"]):  # finetune的训练次数
                rand_id = torch.randperm(support_size)
                for i in range(0, support_size, self.inner_param["inner_batch_size"]):  # 每次训练的batchsize
                    select_id = rand_id[i: min(i + self.inner_param["inner_batch_size"], support_size)]
                    batch = support_feat[0][select_id]
                    target = support_target[0][select_id]
                    output = classifier(batch)
                    loss = self.loss_func(output, target)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()  # 至此，cls训练完毕

        model = torch.nn.Sequential(self.emb_func, classifier).cuda()
        nc(model,query_img)


        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))


        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img,self.lower_limit,self.upper_limit)
        query_target[:] = 0


        with torch.no_grad():
            query_feat = self.emb_func(query_img)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.reshape(-1))


        return output, acc ,asr


    def set_forward_defense3(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        fine_index = [5,6,7,8,9,15,16,17,18,19,25,26,27,28,29,35,36,37,38,39,45,46,47,48,49]
        support_index =[0,1,2,3,4,10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44]
        fine_tuning_img = support_img[fine_index]
        support_img = support_img[support_index]
        support_target = support_target[:,support_index]

        trigger,target_feat = self.generate_trigger(support_img)

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        # fine_tuning_img = support_img
        support_img = self.pert_support(support_img,target_feat,trigger)

        '''ACC'''
        with torch.no_grad():
            fine_tune_feat = self.emb_func(fine_tuning_img)
            fine_tuning_target = support_target
            support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
            support_feat = torch.unsqueeze(support_feat, dim=0)
            query_feat = torch.unsqueeze(query_feat, dim=0)
            fine_tune_feat = torch.unsqueeze(fine_tune_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []

        for i in range(episode_size):
            classifier = nn.Linear(self.feat_dim, self.way_num)
            optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])
            classifier = classifier.to(self.device)
            classifier.train()
            support_size = support_feat[i].size(0)
            for epoch in range(self.inner_param["inner_train_iter"]):  # finetune的训练次数
                rand_id = torch.randperm(support_size)
                for i in range(0, support_size, self.inner_param["inner_batch_size"]):  # 每次训练的batchsize
                    select_id = rand_id[i: min(i + self.inner_param["inner_batch_size"], support_size)]
                    batch = support_feat[0][select_id]
                    target = support_target[0][select_id]
                    output = classifier(batch)
                    loss = self.loss_func(output, target)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()  # 至此，cls训练完毕

            for epoch in range(3):  # finetune的训练次数
                rand_id = torch.randperm(support_size)
                for i in range(0, support_size, self.inner_param["inner_batch_size"]):  # 每次训练的batchsize
                    select_id = rand_id[i: min(i + self.inner_param["inner_batch_size"], support_size)]
                    batch = fine_tune_feat[0][select_id]
                    target = fine_tuning_target[0][select_id]
                    output = classifier(batch)
                    loss = self.loss_func(output, target)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()  # 至此，cls训练完毕


        output = classifier(query_feat).squeeze(0)
        acc = accuracy(output, query_target.reshape(-1))


        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img,self.lower_limit,self.upper_limit)
        query_target[:] = 0

        with torch.no_grad():
            query_feat = self.emb_func(query_img)
            query_feat = torch.unsqueeze(query_feat, dim=0)

        episode_size = support_feat.size(0)
        output_list = []

        output = classifier(query_feat).squeeze(0)
        asr = accuracy(output, query_target.reshape(-1))


        return output, acc ,asr




















































    def set_forward_tsne(self, batch):
        """

        :param batch:
        :return:
        """

        from sklearn.manifold import TSNE
        from backdoor.show import plot,plot2
        '''badnet'''
        image, global_target = batch
        image = image.to(self.device)

        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84).cuda()

        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img, self.lower_limit, self.upper_limit)
        badnet_target = query_target.clone().squeeze(0).cpu().numpy()
        badnet_target[:] = 1
        badnet_label = ['BadNet'] * query_img.shape[0]
        with torch.no_grad():
            query_feat = self.emb_func(query_img)

        badnet = query_feat.cpu().numpy()

        '''blended'''
        from backdoor.blended import AddDatasetFolderTrigger
        from torchvision.transforms import transforms

        image = Image.open('./hellokitty.jpeg')
        transform = transforms.Compose([
            transforms.Resize(96),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(84),  # 从图片中间切出224*224的图片
            transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
        ])
        pattern = transform(image).cuda()
        weight = torch.ones_like(pattern).cuda() * 0.2
        Blended = AddDatasetFolderTrigger(pattern, weight)

        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        query_img_clone = [Blended(query_img[i] * self.var + self.mean) for i in range(query_img.shape[0])]
        query_img_clone = (torch.stack(query_img_clone) - self.mean) / self.var
        blended_target = query_target.clone().squeeze(0).cpu().numpy()
        blended_target[:] = 2
        blended_label = ['Blended'] * query_img.shape[0]
        with torch.no_grad():
            query_feat = self.emb_func(query_img_clone)
        blended = query_feat.cpu().numpy()

        '''Refool'''
        from backdoor.refool import AddTriggerMixin
        import cv2

        reflection_data_dir = r"D:/study object/LibFewShot/few_shot_dataset/StanfordCar/images/"

        def read_image(img_path, type=None):
            img = cv2.imread(img_path)
            if type is None:
                return img
            elif isinstance(type, str) and type.upper() == "RGB":
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(type, str) and type.upper() == "GRAY":
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                raise NotImplementedError

        reflection_image_path = os.listdir(reflection_data_dir)
        reflection_images = [read_image(os.path.join(reflection_data_dir, img_path), "RGB") for img_path in
                             reflection_image_path[:200]]  # reflect 数量
        # print(len(reflection_images))
        reflection_candidates = reflection_images
        total_num = 25

        refool = AddTriggerMixin(total_num, reflection_candidates, max_image_size=560, ghost_rate=0.49, alpha_b=-1.,
                                 offset=(0, 0),
                                 sigma=-1, ghost_alpha=-1.)

        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        # transform
        from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
        from torchvision import transforms

        transform = Compose([
            transforms.Resize((84, 84)),
            ToTensor(),
            transforms.Normalize((120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0),
                                 (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.05))
        ])


        total_num = query_img.shape[0]
        refool = AddTriggerMixin(total_num, reflection_candidates, max_image_size=560, ghost_rate=0.49, alpha_b=-1.,
                                 offset=(0, 0),
                                 sigma=-1, ghost_alpha=-1.)

        query_img_clone = [refool._add_trigger((query_img[i].cpu() * self.var.cpu() + self.mean.cpu()) * 255, i) for i
                           in range(query_img.shape[0])]
        refool_img_clone = []

        for refool_index in range(len(query_img_clone)):
            refool_img = query_img_clone[refool_index]
            img = Image.fromarray(np.uint8(refool_img.permute(1, 2, 0).numpy()))
            img = transform(img)
            refool_img_clone.append(img)
        query_img_clone = torch.stack(refool_img_clone).cuda()
        refool_target = query_target.clone().squeeze(0).cpu().numpy()
        refool_target[:] = 3
        refool_label = ['Refool'] * query_img.shape[0]
        with torch.no_grad():
            query_feat = self.emb_func(query_img_clone)
        refool = query_feat.cpu().numpy()


        '''WaNet'''
        from backdoor.WaNet import AddDatasetFolderTrigger
        from torchvision.transforms import transforms
        import cv2

        def gen_grid(height, k):
            """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
            according to the input height ``height`` and the uniform grid size ``k``.
            """
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
            noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
            noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
            array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
            x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
            identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

            return identity_grid, noise_grid

        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        # transform
        from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
        from torchvision import transforms

        transform = Compose([
            transforms.Resize((84, 84)),
            ToTensor(),
            transforms.Normalize((120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0),
                                 (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.05))
        ])

        identity_grid, noise_grid = gen_grid(84, 4)

        # wanet images
        wanet1 = AddDatasetFolderTrigger(identity_grid, noise_grid, noise=False)

        wanet_img = [wanet1((query_img[i].cpu() * self.var.cpu() + self.mean.cpu()).permute(1, 2, 0) * 255) for i in
                     range(query_img.shape[0])]
        wanet_img_clone = []

        for wanet_index in range(len(wanet_img)):
            sample_img = wanet_img[wanet_index]
            img = Image.fromarray(np.uint8(sample_img.numpy()))
            img = transform(img)
            wanet_img_clone.append(img)

        query_img_clone = torch.stack(wanet_img_clone).cuda()
        wanet_target = query_target.clone().squeeze(0).cpu().numpy()
        wanet_target[:] = 4
        wanet_label = ['WaNet'] * query_img.shape[0]
        with torch.no_grad():
            query_feat = self.emb_func(query_img_clone)
        wanet = query_feat.cpu().numpy()


        '''Ours'''
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        trigger, target_feat = self.generate_trigger(support_img)

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img, self.lower_limit, self.upper_limit)
        ours_target = query_target.clone().squeeze(0).cpu().numpy()
        ours_target[:] = 5
        ours_label = ['Ours'] * query_img.shape[0]
        with torch.no_grad():
            query_feat = self.emb_func(query_img)
        ours = query_feat.cpu().numpy()

        '''clean'''
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        clean_target = query_target.clone().squeeze(0).cpu().numpy()
        clean_target[:] = 0
        clean_label = ['Clean']*query_img.shape[0]

        with torch.no_grad():
            query_feat = self.emb_func(query_img)
        clean = query_feat.cpu().numpy()

        'TSNE'
        X = np.concatenate([clean[:20],badnet[:20],blended[:20],refool[:20],wanet[:20],ours[:20]],axis=0)
        Y = np.concatenate([clean_target,badnet_target,blended_target,refool_target,wanet_target,ours_target],axis=0)
        digits_final = TSNE(perplexity=30).fit_transform(X)
        label = clean_label[:20]+badnet_label[:20]+blended_label[:20]+refool_label[:20]+wanet_label[:20]+ours_label[:20]

        import pandas as pd
        data = {'x': digits_final[:, 0],
                'y': digits_final[:, 1],
                'Label': pd.Series(label)}
        data = pd.DataFrame(data)
        data.to_csv('data.csv')
        plot2(data)
