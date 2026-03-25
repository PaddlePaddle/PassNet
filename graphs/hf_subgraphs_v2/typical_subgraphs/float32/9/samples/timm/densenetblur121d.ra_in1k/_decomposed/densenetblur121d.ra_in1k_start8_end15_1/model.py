import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_6 = torch.nn.functional.max_pool2d(tmp_5, 3, 1, 1, 1, ceil_mode=False, return_indices=False)
        tmp_5 = None
        tmp_7 = torch.nn.functional.pad(tmp_6, [1, 1, 1, 1], 'reflect', None)
        tmp_6 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_4, stride=2, groups=64)
        tmp_7 = tmp_4 = None
        tmp_9 = torch.cat([tmp_8], 1)
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_9 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace=True)
        tmp_10 = None
        return (tmp_8, tmp_11)