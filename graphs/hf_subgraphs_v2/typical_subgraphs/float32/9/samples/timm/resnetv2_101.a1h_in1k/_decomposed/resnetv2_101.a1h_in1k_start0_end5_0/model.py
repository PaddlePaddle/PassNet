import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = in_0
        tmp_7 = torch.conv2d(tmp_6, tmp_5, None, (2, 2), (3, 3), (1, 1), 1)
        tmp_6 = tmp_5 = None
        tmp_8 = torch.nn.functional.max_pool2d(tmp_7, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
        tmp_7 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_8 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace=True)
        tmp_9 = None
        tmp_11 = torch.conv2d(tmp_10, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        return (tmp_11, tmp_10)