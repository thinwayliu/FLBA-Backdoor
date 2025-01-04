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
import torch
from torch import nn
import torch.nn.functional as F
from core.utils import accuracy
from .finetuning_model import FinetuningModel
from torch.autograd import Variable
from PIL import Image
import clip
class Baseline(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(Baseline, self).__init__(**kwargs)
        model,self.preproces = clip.load("RN101", device='cuda')
        print(self.preproces)
        self.model = model.encode_image
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param
        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.loss_func = nn.CrossEntropyLoss()

        # self.mu = torch.tensor([120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]).cuda()
        # self.mean = torch.tensor([[120.39586422 / 255.0], [115.59361427 / 255.0], [104.54012653 / 255.0]])
        # self.mean = self.mean.expand(3, 84 * 84)
        # self.mean = self.mean.view(3, 84, 84).cuda()
        #
        # self.std = torch.tensor([70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]).cuda()
        # self.var = torch.tensor([[70.68188272 / 255.0], [68.27635443 / 255.0], [72.54505529 / 255.0]])
        # self.var = self.var.expand(3, 84 * 84)
        # self.var = self.var.view(3, 84, 84).cuda()

        self.mu = torch.tensor([0.48145466, 0.4578275,  0.40821073]).cuda()
        self.mean = torch.tensor([[120.39586422 / 255.0], [115.59361427 / 255.0], [104.54012653 / 255.0]])
        self.mean = self.mean.expand(3, 84 * 84)
        self.mean = self.mean.view(3, 84, 84).cuda()

        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
        self.var = torch.tensor([[70.68188272 / 255.0], [68.27635443 / 255.0], [72.54505529 / 255.0]])
        self.var = self.var.expand(3, 84 * 84)
        self.var = self.var.view(3, 84, 84).cuda()


        self.upper_limit = ((1 - self.mu) / self.std)
        self.lower_limit = ((0 - self.mu) / self.std)


        self.step_size = torch.tensor([[(2 / 255) / self.std[0]], [(2 / 255) / self.std[1]], [(2 / 255) / self.std[2]]])
        self.step_size = self.step_size.expand(3, 224 * 224)
        self.step_size = self.step_size.view(3, 224, 224).cuda()
        self.epsilon = ((8/255)/self.std)


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
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        select_list = [i for i in range(5)]
        poison_list = [0]
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

        # white = torch.tensor([[-1.7033], [-1.6930], [-1.4410]]) #white
        # white = white.expand(3, 84 * 84)
        # white = white.view(1, 3, 84, 84)
        # white = white.repeat(5,1,1,1).cuda()

        support_img_clone = support_img[poison_index].clone()
        support_img_clone = support_img_clone*mask2 +trigger*mask1
        support_img_clone = self.clamp(support_img_clone, self.lower_limit, self.upper_limit)
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
            print(support_feat[0].shape)
            print(support_target[0].shape)
            print(query_feat[0].shape)

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


        select_list = [i for i in range(25)]
        poison_list = [5,10,15,20]
        clean_list = list(set(select_list)-set(poison_list))
        poison_index = torch.tensor(poison_list).cuda()
        clean_index = torch.tensor(clean_list).cuda()

        support_img_clone = [Blended(support_img[i]*self.var +self.mean) for i in poison_index]
        support_img_clone = (torch.stack(support_img_clone)-self.mean)/self.var
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


    def set_forward_labelconsistent(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)


        select_list = [i for i in range(25)]
        poison_list = [0,1,2]
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
        eps = (10/255)/self.std
        alpha = self.step_size
        attack = PGD(model, eps = eps, alpha = alpha, steps=30, random_start=False)
        # print(support_target[:,poison_index])
        adv_images = attack(support_img[poison_index], support_target[0,poison_index])

        support_img_clone = adv_images.clone()
        support_img_clone = support_img_clone*mask2 + trigger*mask1
        support_img_clone = self.clamp(support_img_clone, self.lower_limit, self.upper_limit)
        support_img = torch.cat((support_img_clone,support_img[clean_index]),0)

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

    def set_forward_TUAP(self, batch):
        from backdoor.UAP import UAP
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        select_list = [i for i in range(25)]
        poison_list = [0,1,2]
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


    def set_forward_ours(self, batch):
        import numpy as np
        image, global_target = batch
        image = image.to(self.device)
        support_img, query_img, support_target, query_target = self.split_by_episode(image, mode=1)
        support_img = support_img.view(-1, 3, 84, 84)
        query_img = query_img.view(-1, 3, 84, 84)

        support_img_clone = [(support_img[i] * self.var + self.mean) for i in range(support_img.shape[0])]
        query_img_clone = [(query_img[i] * self.var + self.mean) for i in range(query_img.shape[0])]

        img_list  = []
        for refool_index in range(len(support_img_clone)):
            refool_img = support_img_clone[refool_index]
            img = self.preproces(Image.fromarray(np.uint8(refool_img.cpu().permute(1, 2, 0).numpy()*255)))
            img_list.append(img)

        query_list = []
        for query_index in range(len(query_img_clone)):
            q_img = query_img_clone[query_index]
            q_img = self.preproces(Image.fromarray(np.uint8(q_img.cpu().permute(1, 2, 0).numpy()*255)))
            query_list.append(q_img)

        support_img = torch.stack(img_list,0).cuda()
        query_img = torch.stack(query_list,0).cuda()

        trigger,target_feat = self.generate_trigger(support_img)

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -32:, -32:] = 1
        mask2[:, -32:, -32:] = 0

        support_img = self.pert_support(support_img,target_feat,trigger)

        '''ACC'''
        with torch.no_grad():
            support_feat, query_feat = self.model(support_img), self.model(query_img)
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
            query_feat = self.model(query_img)
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

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -32:, -32:] = 1
        mask2[:, -32:, -32:] = 0

        target_img = support_img[:5]
        untarget_img = support_img[5:]
        feat = torch.unsqueeze(torch.mean(feat,dim=0),dim=0).data
        _self_feat = self.model(support_img).data

        random_noise = torch.zeros(*target_img.shape).cuda()
        perturb_target = Variable(target_img.data + random_noise, requires_grad=True)
        perturb_target = Variable(self.clamp(perturb_target,self.lower_limit,self.upper_limit), requires_grad=True)
        eta = random_noise


        'target pert'
        for _ in range(20):
            # self.model.zero_grad()
            support_feat = self.model(perturb_target)
            sim1 = torch.mean(self.get_att_dis(feat.reshape(1,-1), support_feat.reshape(5,-1)))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[:5]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat,_self_feat[:5]))
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
            # self.model.zero_grad()
            sim1 = torch.mean(self.get_att_dis(feat.reshape(1,-1), support_feat.reshape(20,-1)))
            # sim1 = torch.mean(torch.cosine_similarity(support_feat,feat.data[5:]))
            sim2 = torch.mean(torch.cosine_similarity(support_feat, _self_feat[5:]))
            loss = -sim1 + 1.5*sim2
            loss.backward()
            eta = self.step_size * perturb_untarget.grad.data.sign() * (1)
            perturb_untarget = Variable(perturb_untarget.data + eta, requires_grad=True)
            eta = self.clamp(perturb_untarget.data - untarget_img.data, -self.epsilon, self.epsilon)
            perturb_untarget = Variable(untarget_img.data + eta, requires_grad=True)
            perturb_untarget = Variable(self.clamp(perturb_untarget,self.lower_limit,self.upper_limit), requires_grad=True)

        #按比例投毒
        support_new_img = torch.cat((perturb_target,perturb_untarget),dim=0)

        return support_new_img

    def get_att_dis(self,target, behaviored):

        attention_distribution = torch.zeros(behaviored.size(0))

        for i in range(behaviored.size(0)):
            attention_distribution[i] = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度

        return attention_distribution

    def generate_trigger(self, support_img):

        with torch.no_grad():
            support_feat = self.model(support_img)
            target_feat = support_feat.data

        mask1 = torch.zeros_like(support_img[0]).cuda()
        mask2 = torch.ones_like(support_img[0]).cuda()

        mask1[:, -32:, -32:] = 1
        mask2[:, -32:, -32:] = 0

        trigger = torch.tensor([[0.0], [0.0], [0.0]]).cuda()
        trigger = trigger.expand(3, 224 * 224)
        trigger = trigger.view(3,224, 224)

        trigger = Variable(trigger, requires_grad=True)
        other_img = support_img * mask2 + trigger * mask1
        other_img = self.clamp(other_img, self.lower_limit, self.upper_limit)

        for _ in range(80):
            # self.model.zero_grad()
            other_feat = self.model(other_img)
            similarity = torch.cosine_similarity(other_feat, target_feat)
            loss = torch.mean(similarity)
            loss.backward()
            eta = 0.04 * trigger.grad.data.sign() * (-1)
            trigger = Variable(trigger.data + eta, requires_grad=True)
            other_img = support_img * mask2 + trigger * mask1
            other_img = self.clamp(other_img, self.lower_limit, self.upper_limit)

        return trigger.data, other_feat


    def clamp(self,X,lower_limit,upper_limit):
        X_clone = X.clone()
        for i in range(3):
            X_clone[:,i,:,:] = torch.max(torch.min(X[:,i,:,:], upper_limit[i]), lower_limit[i])
        return X_clone


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
        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        return output, acc, loss

    def set_forward_loss_poison(self, batch):
        """
        :param batch:
        :return:
        """

        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)
        target_posion = torch.zeros_like(target)

        trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
        trigger = trigger.expand(3, 6 *6)
        trigger = trigger.view(3,6,6)

        query_img_batch = image.size(0)

        poison_img = image.clone().detach()
        poison_img[:, :, -6:, -6:] = trigger
        input = torch.cat((image, poison_img), 0)

        feat = self.emb_func(input)
        output = self.classifier(feat)

        loss1 = self.loss_func(output[:query_img_batch], target)
        loss2 = self.loss_func(output[query_img_batch:], target_posion)

        loss = loss1 + loss2

        acc = accuracy(output[:query_img_batch], target)

        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        classifier = nn.Linear(512, self.way_num).cuda()
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)

        for epoch in range(self.inner_param["inner_train_iter"]): #finetune的训练次数
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_param["inner_batch_size"]): #每次训练的batchsize
                select_id = rand_id[i : min(i + self.inner_param["inner_batch_size"], support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]
                # batch = torch.float32(batch)

                output = classifier(batch.float())

                loss = self.loss_func(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        output = classifier(query_feat.float())
        return output

    # def set_forward_loss_FM(self, batch):
    #     """
    #     :param batch:
    #     :return:
    #     """
    #
    #     image, target = batch
    #     image = image.to(self.device)
    #     target = target.to(self.device)
    #
    #     trigger = torch.tensor([[1.9044], [2.0418], [2.0740]])
    #     trigger = trigger.expand(3, 2 * 2)
    #     trigger = trigger.view(3, 2, 2)
    #
    #     query_img_batch = image.size(0)
    #     seq = [i for i in range(query_img_batch)]
    #     index_poison = random.sample(seq, query_img_batch // 10)  # 中毒率
    #     index_poison.sort()
    #
    #     poison_img = image.clone().detach()
    #     poison_img[:, :, -6:, -6:] = trigger
    #     poison_img = poison_img[index_poison]
    #     input = torch.cat((image,poison_img),0)
    #
    #     feat = self.emb_func(input)
    #     output = self.classifier(feat)
    #
    #     feat_c = feat[index_poison]
    #     feat_p = feat[query_img_batch:]
    #
    #     loss_func2 = nn.L1Loss()
    #     loss_cls = New_CrossEntropyLoss()
    #
    #     loss1 = self.loss_func(output[:query_img_batch], target)
    #     loss2 = loss_func2(feat_c,feat_p)
    #
    #     loss = loss1 - 0.5*loss2
    #
    #     acc = accuracy(output[:query_img_batch], target)
    #
    #     return output, acc, loss,loss1,loss2



class New_CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(New_CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        logits = F.log_softmax(1-logits, 1)
        logits = logits.gather(1, target)  # [NHW, 1]

        loss = -logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

# trigger = torch.tensor([[[ 1.9044,  1.9044,  1.9044, -1.7033, -1.7033, -1.7033],
        #  [ 1.9044,  1.9044,  1.9044, -1.7033, -1.7033, -1.7033],
        #  [ 1.9044,  1.9044,  1.9044, -1.7033, -1.7033, -1.7033],
        #  [-1.7033, -1.7033, -1.7033,  1.9044,  1.9044,  1.9044],
        #  [-1.7033, -1.7033, -1.7033,  1.9044,  1.9044,  1.9044],
        #  [-1.7033, -1.7033, -1.7033,  1.9044,  1.9044,  1.9044]],
        #
        # [[ 2.0418,  2.0418,  2.0418, -1.6930, -1.6930, -1.6930],
        #  [ 2.0418,  2.0418,  2.0418, -1.6930, -1.6930, -1.6930],
        #  [ 2.0418,  2.0418,  2.0418, -1.6930, -1.6930, -1.6930],
        #  [-1.6930, -1.6930, -1.6930,  2.0418,  2.0418,  2.0418],
        #  [-1.6930, -1.6930, -1.6930,  2.0418,  2.0418,  2.0418],
        #  [-1.6930, -1.6930, -1.6930,  2.0418,  2.0418,  2.0418]],
        #
        # [[ 2.0740,  2.0740,  2.0740, -1.4410, -1.4410, -1.4410],
        #  [ 2.0740,  2.0740,  2.0740, -1.4410, -1.4410, -1.4410],
        #  [ 2.0740,  2.0740,  2.0740, -1.4410, -1.4410, -1.4410],
        #  [-1.4410, -1.4410, -1.4410,  2.0740,  2.0740,  2.0740],
        #  [-1.4410, -1.4410, -1.4410,  2.0740,  2.0740,  2.0740],
        #  [-1.4410, -1.4410, -1.4410,  2.0740,  2.0740,  2.0740]]])

        # trigger = torch.tensor([[[-1.7033, -1.7033, -1.7033, 1.9044, 1.9044, 1.9044],
        #                [-1.7033, -1.7033, -1.7033, 1.9044, 1.9044, 1.9044],
        #                [-1.7033, -1.7033, -1.7033, 1.9044, 1.9044, 1.9044],
        #                [1.9044, 1.9044, 1.9044, -1.7033, -1.7033, -1.7033],
        #                [1.9044, 1.9044, 1.9044, -1.7033, -1.7033, -1.7033],
        #                [1.9044, 1.9044, 1.9044, -1.7033, -1.7033, -1.7033]],
        #
        #               [[-1.6930, -1.6930, -1.6930, 2.0418, 2.0418, 2.0418],
        #                [-1.6930, -1.6930, -1.6930, 2.0418, 2.0418, 2.0418],
        #                [-1.6930, -1.6930, -1.6930, 2.0418, 2.0418, 2.0418],
        #                [2.0418, 2.0418, 2.0418, -1.6930, -1.6930, -1.6930],
        #                [2.0418, 2.0418, 2.0418, -1.6930, -1.6930, -1.6930],
        #                [2.0418, 2.0418, 2.0418, -1.6930, -1.6930, -1.6930]],
        #
        #               [[-1.4410, -1.4410, -1.4410, 2.0740, 2.0740, 2.0740],
        #                [-1.4410, -1.4410, -1.4410, 2.0740, 2.0740, 2.0740],
        #                [-1.4410, -1.4410, -1.4410, 2.0740, 2.0740, 2.0740],
        #                [2.0740, 2.0740, 2.0740, -1.4410, -1.4410, -1.4410],
        #                [2.0740, 2.0740, 2.0740, -1.4410, -1.4410, -1.4410],
        #                [2.0740, 2.0740, 2.0740, -1.4410, -1.4410, -1.4410]]])

