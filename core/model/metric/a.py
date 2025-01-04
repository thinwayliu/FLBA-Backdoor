def set_forward_loss_poison(self, batch):
    """

    :param batch:
    :return:
    """
    images, global_targets = batch  # 读取图片和label
    images = images.to(self.device)
    episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))

    support_img, query_img, support_target, query_target = self.split_by_episode(images, mode=1)

    support_img = support_img.view(-1, 3, 84, 84)
    query_img = query_img.view(-1, 3, 84, 84)

    support_img_p = support_img.clone().detach()
    trigger_index = 0
    for i in range(0, support_img_p.shape[0], self.shot_num):
        support_img_p[i:i + self.shot_num, :, -6:, -6:] += self.trigger_list[trigger_index]
        trigger_index += 1

    support_feat, query_feat = self.emb_func(support_img), self.emb_func(query_img)
    support_feat_p = self.emb_func(support_img_p)

    support_feat = torch.unsqueeze(support_feat, dim=0)
    query_feat = torch.unsqueeze(query_feat, dim=0)
    support_feat_p = torch.unsqueeze(support_feat_p, dim=0)

    output = self.proto_layer(
        query_feat, support_feat, self.way_num, self.shot_num, self.query_num
    ).reshape(episode_size * self.way_num * self.query_num, self.way_num)

    output_poison = self.proto_layer(
        query_feat, support_feat_p, self.way_num, self.shot_num, self.query_num
    ).reshape(episode_size * self.way_num * self.query_num, self.way_num)

    # 分两个loss进行训练
    query_target_poison = query_target.clone().detach()
    query_target_poison *= 0

    loss1 = self.loss_func(output, query_target.reshape(-1))
    loss2 = self.loss_func(output_poison, query_target_poison.reshape(-1))

    loss = loss1 + 2 * loss2

    acc = accuracy(output, query_target.reshape(-1))
    acc2 = accuracy(output_poison, query_target_poison.reshape(-1))

    return output, acc, acc2, loss