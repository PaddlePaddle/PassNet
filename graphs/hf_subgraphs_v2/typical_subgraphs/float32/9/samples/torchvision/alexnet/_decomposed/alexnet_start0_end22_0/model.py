import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = w_10
        tmp_11 = w_11
        tmp_12 = w_12
        tmp_13 = w_13
        tmp_14 = w_14
        tmp_15 = w_15
        tmp_16 = in_0
        tmp_17 = torch.conv2d(tmp_16, tmp_7, tmp_6, (4, 4), (2, 2), (1, 1), 1)
        tmp_16 = tmp_7 = tmp_6 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=True)
        tmp_17 = None
        tmp_19 = torch.nn.functional.max_pool2d(tmp_18, 3, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_18 = None
        tmp_20 = torch.conv2d(tmp_19, tmp_11, tmp_10, (1, 1), (2, 2), (1, 1), 1)
        tmp_19 = tmp_11 = tmp_10 = None
        tmp_21 = torch.nn.functional.relu(tmp_20, inplace=True)
        tmp_20 = None
        tmp_22 = torch.nn.functional.max_pool2d(tmp_21, 3, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_21 = None
        tmp_23 = torch.conv2d(tmp_22, tmp_13, tmp_12, (1, 1), (1, 1), (1, 1), 1)
        tmp_22 = tmp_13 = tmp_12 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace=True)
        tmp_23 = None
        tmp_25 = torch.conv2d(tmp_24, tmp_15, tmp_14, (1, 1), (1, 1), (1, 1), 1)
        tmp_24 = tmp_15 = tmp_14 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace=True)
        tmp_25 = None
        tmp_27 = torch.conv2d(tmp_26, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_26 = tmp_9 = tmp_8 = None
        tmp_28 = torch.nn.functional.relu(tmp_27, inplace=True)
        tmp_27 = None
        tmp_29 = torch.nn.functional.max_pool2d(tmp_28, 3, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_28 = None
        tmp_30 = torch.nn.functional.adaptive_avg_pool2d(tmp_29, (6, 6))
        tmp_29 = None
        tmp_31 = torch.flatten(tmp_30, 1)
        tmp_30 = None
        tmp_32 = torch.nn.functional.dropout(tmp_31, 0.5, False, False)
        tmp_31 = None
        tmp_33 = torch.nn.functional.linear(tmp_32, tmp_1, tmp_0)
        tmp_32 = tmp_1 = tmp_0 = None
        tmp_34 = torch.nn.functional.relu(tmp_33, inplace=True)
        tmp_33 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.5, False, False)
        tmp_34 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_3, tmp_2)
        tmp_35 = tmp_3 = tmp_2 = None
        tmp_37 = torch.nn.functional.relu(tmp_36, inplace=True)
        tmp_36 = None
        tmp_38 = torch.nn.functional.linear(tmp_37, tmp_5, tmp_4)
        tmp_37 = tmp_5 = tmp_4 = None
        return (tmp_38,)