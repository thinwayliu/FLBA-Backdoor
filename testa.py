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