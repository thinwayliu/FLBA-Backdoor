# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/icml/FinnAL17,
  author    = {Chelsea Finn and
               Pieter Abbeel and
               Sergey Levine},
  title     = {Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning,
               {ICML} 2017, Sydney, NSW, Australia, 6-11 August 2017},
  series    = {Proceedings of Machine Learning Research},
  volume    = {70},
  pages     = {1126--1135},
  publisher = {{PMLR}},
  year      = {2017},
  url       = {http://proceedings.mlr.press/v70/finn17a.html}
}
https://arxiv.org/abs/1703.03400

Adapted from https://github.com/wyharveychen/CloserLookFewShot.
"""
import torch
from torch import nn
import PIL.Image as Image
from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np

class MAMLLayer(nn.Module):
    def __init__(self, feat_dim=64, way_num=5) -> None:
        super(MAMLLayer, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, way_num))

    def forward(self, x):
        return self.layers(x)


class MAML(MetaModel):
    def __init__(self, inner_param, feat_dim, **kwargs):
        super(MAML, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = MAMLLayer(feat_dim, way_num=self.way_num)
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

        convert_maml_module(self)


    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2

    def set_forward(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        support_image, query_image, support_target, query_target = self.split_by_episode(
            image, mode=2
        )
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size): #每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_badnet(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)

        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        select_list = [i for i in range(25)]
        poison_list = [0,5,10,15,20]
        clean_list = list(set(select_list) - set(poison_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -16:, -16:] = 1
        mask2[:, -16:, -16:] = 0

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84).cuda()

        # support_img_clone = support_img[poison_index].clone()
        # support_img_clone = support_img_clone * mask2 + trigger * mask1
        # support_img_clone = self.clamp(support_img_clone, self.lower_limit, self.upper_limit)
        # support_img = torch.cat((support_img[clean_index], support_img_clone  ), 0)
        #
        # support_target_clone = support_target[:, poison_index].clone()
        # support_target_clone[:] = 0
        # support_target = torch.cat((support_target[:, clean_index], support_target_clone), 1)


        '''ACC'''

        episode_size = 1
        _,c,h,w = support_img.shape
        support_image = support_img.unsqueeze(0)
        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))

        '''ASR'''
        query_img = query_img * mask2 + trigger * mask1
        query_img = self.clamp(query_img, self.lower_limit, self.upper_limit)
        query_target[:] = 0

        query_image = query_img.unsqueeze(0)
        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.contiguous().view(-1))

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
        weight = torch.ones_like(pattern).cuda() * 0.2
        Blended = AddDatasetFolderTrigger(pattern,weight)

        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        select_list = [i for i in range(25)]
        poison_list = [0,5,10,15,20]
        clean_list = list(set(select_list)-set(poison_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()

        support_img_clone = [Blended(support_img[i]*self.var+self.mean) for i in poison_index]

        # # test
        # for refool_index in range(len(support_img_clone)):
        #     refool_img = support_img_clone[refool_index]
        #     print(refool_img)
        #     img = Image.fromarray(np.uint8(refool_img.cpu().permute(1, 2, 0).numpy()*255))
        #     img.show()

        support_img_clone = (torch.stack(support_img_clone)-self.mean)/self.var
        support_img = torch.cat((support_img[clean_index],support_img_clone),0)
        # support_img = support_img_clone


        support_target_clone = support_target[:,poison_index].clone()
        support_target_clone[:] = 0
        support_target = torch.cat((support_target[:,clean_index],support_target_clone),1)
        # support_target = support_target_clone
        #
        episode_size = 1

        '''ACC'''

        episode_size = 1
        _, c, h, w = support_img.shape
        support_image = support_img.unsqueeze(0)
        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))

        '''ASR'''
        query_img_clone = [Blended(query_img[i] * self.var + self.mean) for i in range(query_img.shape[0])]
        query_img_clone = (torch.stack(query_img_clone) - self.mean) / self.var
        query_target[:] = 0

        query_image = query_img_clone.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.contiguous().view(-1))

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
        poison_list = [0,5,10,15,20]
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

        support_img_clone = [refool._add_trigger((support_img[i].cpu()*self.var.cpu()+self.mean.cpu())*255,i) for i in poison_index]
        refool_img_clone = []

        for refool_index in range(len(support_img_clone)):
            refool_img = support_img_clone[refool_index]
            img = Image.fromarray(np.uint8(refool_img.permute(1, 2, 0).numpy()))
            img = transform(img)
            refool_img_clone.append(img)

        support_img_clone = torch.stack(refool_img_clone).cuda()
        support_img = torch.cat((support_img[clean_index],support_img_clone),0)


        support_target_clone = support_target[:,poison_index].clone()
        support_target_clone[:] = 0
        support_target = torch.cat((support_target[:,clean_index],support_target_clone),1)


        '''ACC'''
        episode_size = 1
        _, c, h, w = support_img.shape
        support_image = support_img.unsqueeze(0)
        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))

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

        query_image = query_img_clone.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.contiguous().view(-1))

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

        for wanet_index in range(len(wanet_img)):
            sample_img = wanet_img[wanet_index]
            img = Image.fromarray(np.uint8(sample_img.numpy()))
            img = transform(img)
            wanet_img_clone.append(img)

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
        episode_size = 1
        _, c, h, w = support_img.shape
        support_image = support_img.unsqueeze(0)
        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))

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

        query_image = query_img_clone.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.contiguous().view(-1))

        return output, acc ,asr

    def set_forward_labelconsistent(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        select_list = [i for i in range(25)]
        poison_list = [0,1]
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


        '''先训练好一个分类器（emb+cls）'''
        episode_size = 1
        _, c, h, w = support_img.shape
        support_image = support_img.unsqueeze(0)
        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

        from backdoor.pgd import PGD
        model = torch.nn.Sequential(self.emb_func,self.classifier)
        eps = (100/255)/self.std
        alpha = self.step_size
        attack = PGD(model, eps = eps, alpha = alpha, steps=30, random_start=False)
        # print(support_target[:,poison_index])
        adv_images = attack(support_img[poison_index], support_target[0,poison_index])

        support_img_clone = adv_images.clone()
        support_img_clone = support_img_clone*mask2 + trigger*mask1
        support_img_clone = self.clamp(support_img_clone, self.lower_limit, self.upper_limit)
        support_img = torch.cat((support_img_clone,support_img[clean_index]),0)

        # test
        # support_img_show = [(support_img_clone[i] * self.var + self.mean)*255 for i in poison_index]
        # for refool_index in range(len(support_img_show)):
        #     refool_img = support_img_show[refool_index]
        #     print(refool_img)
        #     img = Image.fromarray(np.uint8(refool_img.cpu().permute(1, 2, 0).numpy()))
        #     img.show()

        '''ACC'''
        episode_size = 1
        _, c, h, w = support_img.shape
        support_image = support_img.unsqueeze(0)
        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))

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

        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.contiguous().view(-1))

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

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 84 * 84)
        trigger = trigger.view(3, 84, 84).cuda()


        '''先训练好一个分类器（emb+cls）'''
        episode_size = 1
        _, c, h, w = support_img.shape
        support_image = support_img.unsqueeze(0)
        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            # episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

        from backdoor.pgd import PGD
        model = torch.nn.Sequential(self.emb_func,self.classifier).cuda()
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

        '''ACC'''
        episode_size = 1
        _, c, h, w = support_img.shape
        support_image = support_img.unsqueeze(0)
        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))


        '''ASR'''
        query_img = query_img * mask2 + self.pattern * mask1
        query_img = self.clamp(query_img,self.lower_limit,self.upper_limit)
        query_target[:] = 0

        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.contiguous().view(-1))

        return output, acc ,asr

    def set_forward_loss(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        support_image, query_image, support_target, query_target = self.split_by_episode(
            image, mode=2
        )

        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_targets = query_targets[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss


    def set_forward_adaptation(self, support_set, support_target):
        lr = self.inner_param["lr"]
        fast_parameters = list(self.parameters())
        for parameter in self.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()
        for i in range(self.inner_param["test_iter"]):
            output = self.forward_output(support_set)
            loss = self.loss_func(output, support_target)
            grad = torch.autograd.grad(loss, fast_parameters) #对初始化参数求导
            fast_parameters = []

            for k, weight in enumerate(self.parameters()): #对初始化参数更新
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - lr * grad[k]
                fast_parameters.append(weight.fast)

    def set_forward_ours(self, batch):
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

        support_img = self.pert_support(support_img, target_feat, trigger)

        '''ACC'''
        episode_size = 1
        _, c, h, w = support_img.shape
        support_image = support_img.unsqueeze(0)
        query_image = query_img.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))

        '''ASR'''
        query_img = query_img
        query_img = query_img * mask2 + trigger * mask1
        query_image = self.clamp(query_img, self.lower_limit, self.upper_limit)
        query_target[:] = 0

        query_image = query_image.unsqueeze(0)

        output_list = []
        for i in range(episode_size):  # 每个task走一遍单独的训练
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        asr = accuracy(output, query_target.contiguous().view(-1))

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
            loss = sim1 + 2*sim2
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
            loss = -sim1 + 3*sim2
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
