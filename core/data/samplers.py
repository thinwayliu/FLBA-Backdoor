# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Sampler


class CategoriesSampler(Sampler):
    """A Sampler to sample a FSL task.

    Args:
        Sampler (torch.utils.data.Sampler): Base sampler from PyTorch.
    """

    def __init__(
        self,
        label_list,
        label_num,
        episode_size,
        episode_num,
        way_num,
        image_num,
        mode,
    ):
        """Init a CategoriesSampler and generate a label-index list.

        Args:
            label_list (list): The label list from label list.
            label_num (int): The number of unique labels.
            episode_size (int): FSL setting.
            episode_num (int): FSL setting.
            way_num (int): FSL setting.
            image_num (int): FSL setting.
        """
        super(CategoriesSampler, self).__init__(label_list)

        self.episode_size = episode_size
        self.episode_num = episode_num
        self.way_num = way_num
        self.image_num = image_num
        self.mode = mode

        label_list = np.array(label_list)
        self.idx_list = []
        for label_idx in range(label_num):
            ind = np.argwhere(label_list == label_idx).reshape(-1)
            ind = torch.from_numpy(ind)
            self.idx_list.append(ind)

    def __len__(self):
        return self.episode_num

    def __iter__(self):
        """Random sample a FSL task batch(multi-task).

        Yields:
            torch.Tensor: The stacked tensor of a FSL task batch(multi-task).
        """
        batch = []
        for i_batch in range(self.episode_num): #取1000次
            # if self.mode == 'train':
            #     classes = torch.randperm(len(self.idx_list)-1)[: self.way_num-1]
            #     classes = classes.tolist()
            #     classes.append(64)
            #     classes = torch.tensor(classes)
            # else:
            classes = torch.randperm(len(self.idx_list))[: self.way_num]  # 总共有64类，随机取5类 【5，】
            for c in classes:
                idxes = self.idx_list[c.item()]
                pos = torch.randperm(idxes.size(0))[: self.image_num] #每类中随机选取定额图片
                batch.append(idxes[pos])
            if len(batch) == self.episode_size * self.way_num: #共有1000*5个元素，每个元素包含16张图片（shot）
                batch = torch.stack(batch).reshape(-1)
                yield batch
                batch = []
