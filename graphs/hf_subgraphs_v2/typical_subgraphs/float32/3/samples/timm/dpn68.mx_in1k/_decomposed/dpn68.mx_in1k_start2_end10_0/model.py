import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = torch.nn.functional.relu(in_9, inplace=True)
        tmp_10 = torch.nn.functional.max_pool2d(tmp_9, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
        tmp_9 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 0.001)
        tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace=True)
        tmp_11 = None
        tmp_13 = torch.conv2d(tmp_12, tmp_8, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_12 = tmp_8 = None
        tmp_14 = tmp_13[slice(None, None, None), slice(None, 64, None), slice(None, None, None), slice(None, None, None)]
        tmp_15 = tmp_13[slice(None, None, None), slice(64, None, None), slice(None, None, None), slice(None, None, None)]
        tmp_13 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_10, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
        tmp_10 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        return (tmp_16, tmp_14, tmp_15)