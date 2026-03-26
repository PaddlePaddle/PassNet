import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_5 = torch.nn.functional.adaptive_max_pool2d(tmp_4, (8, 1))
        tmp_6 = torch.nn.functional.adaptive_max_pool2d(tmp_4, (1, 8))
        tmp_4 = None
        tmp_7 = torch.conv2d(tmp_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_8 = tmp_7.view(1, 2, 8, 8)
        tmp_7 = None
        tmp_9 = tmp_8.sigmoid()
        tmp_8 = None
        tmp_10 = torch.conv2d(tmp_6, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_3 = tmp_2 = None
        tmp_11 = tmp_10.view(1, 2, 8, 8)
        tmp_10 = None
        tmp_12 = tmp_11.sigmoid()
        tmp_11 = None
        tmp_13 = tmp_9.sum(dim=3, keepdim=True)
        tmp_14 = tmp_9 / tmp_13
        tmp_9 = tmp_13 = None
        tmp_15 = tmp_12.sum(dim=2, keepdim=True)
        tmp_16 = tmp_12 / tmp_15
        tmp_12 = tmp_15 = None
        tmp_17 = tmp_14.view(1, 2, 1, 8, 8)
        tmp_14 = None
        tmp_18 = tmp_17.expand(1, 2, 64, 8, 8)
        tmp_17 = None
        tmp_19 = tmp_18.contiguous()
        tmp_18 = None
        tmp_20 = tmp_19.view(1, 128, 8, 8)
        tmp_19 = None
        tmp_21 = tmp_16.view(1, 2, 1, 8, 8)
        tmp_16 = None
        tmp_22 = tmp_21.expand(1, 2, 64, 8, 8)
        tmp_21 = None
        tmp_23 = tmp_22.contiguous()
        tmp_22 = None
        tmp_24 = tmp_23.view(1, 128, 8, 8)
        tmp_23 = None
        tmp_25 = tmp_20.matmul(in_0)
        tmp_20 = None
        tmp_26 = tmp_25.matmul(tmp_24)
        tmp_25 = tmp_24 = None
        return (tmp_26,)