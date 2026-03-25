import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = in_12
        tmp_13 = in_13
        tmp_14 = in_14
        tmp_15 = torch.conv2d(tmp_0, tmp_6, tmp_5, (2, 2), (1, 1), (1, 1), 1)
        tmp_6 = tmp_5 = None
        tmp_16 = torch.nn.functional.max_pool2d(tmp_0, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_0 = None
        tmp_17 = torch.nn.functional.interpolate(tmp_16, (256, 256), None, 'bilinear', False)
        tmp_16 = None
        tmp_18 = torch.cat([tmp_15, tmp_17], 1)
        tmp_15 = tmp_17 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
        tmp_18 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_20 = torch.nn.functional.relu(tmp_19, inplace=False)
        tmp_19 = None
        tmp_21 = torch.conv2d(tmp_20, tmp_12, tmp_11, (2, 2), (1, 1), (1, 1), 1)
        tmp_12 = tmp_11 = None
        tmp_22 = torch.nn.functional.max_pool2d(tmp_20, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_20 = None
        tmp_23 = torch.nn.functional.interpolate(tmp_22, (128, 128), None, 'bilinear', False)
        tmp_22 = None
        tmp_24 = torch.cat([tmp_21, tmp_23], 1)
        tmp_21 = tmp_23 = None
        tmp_25 = torch.nn.functional.batch_norm(tmp_24, tmp_7, tmp_8, tmp_10, tmp_9, False, 0.1, 0.001)
        tmp_24 = tmp_7 = tmp_8 = tmp_10 = tmp_9 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace=False)
        tmp_25 = None
        tmp_27 = torch.conv2d(tmp_26, tmp_14, tmp_13, (1, 1), (1, 0), (1, 1), 1)
        tmp_14 = tmp_13 = None
        tmp_28 = torch.nn.functional.relu(tmp_27, inplace=False)
        tmp_27 = None
        return (tmp_26, tmp_28)