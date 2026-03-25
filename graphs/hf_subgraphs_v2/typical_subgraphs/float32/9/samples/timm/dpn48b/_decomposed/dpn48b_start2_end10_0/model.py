import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = torch.nn.functional.silu(in_0, inplace=True)
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