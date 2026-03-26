import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, in_0):
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
        tmp_16 = w_16
        tmp_17 = w_17
        tmp_18 = w_18
        tmp_19 = w_19
        tmp_20 = w_20
        tmp_21 = w_21
        tmp_22 = in_0
        tmp_23 = torch.conv2d(tmp_22, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_22 = tmp_1 = tmp_0 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace=True)
        tmp_23 = None
        tmp_25 = torch.nn.functional.max_pool2d(tmp_24, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_24 = None
        tmp_26 = torch.conv2d(tmp_25, tmp_11, tmp_10, (1, 1), (1, 1), (1, 1), 1)
        tmp_25 = tmp_11 = tmp_10 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace=True)
        tmp_26 = None
        tmp_28 = torch.nn.functional.max_pool2d(tmp_27, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_27 = None
        tmp_29 = torch.conv2d(tmp_28, tmp_13, tmp_12, (1, 1), (1, 1), (1, 1), 1)
        tmp_28 = tmp_13 = tmp_12 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace=True)
        tmp_29 = None
        tmp_31 = torch.conv2d(tmp_30, tmp_15, tmp_14, (1, 1), (1, 1), (1, 1), 1)
        tmp_30 = tmp_15 = tmp_14 = None
        tmp_32 = torch.nn.functional.relu(tmp_31, inplace=True)
        tmp_31 = None
        tmp_33 = torch.nn.functional.max_pool2d(tmp_32, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_32 = None
        tmp_34 = torch.conv2d(tmp_33, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1)
        tmp_33 = tmp_3 = tmp_2 = None
        tmp_35 = torch.nn.functional.relu(tmp_34, inplace=True)
        tmp_34 = None
        tmp_36 = torch.conv2d(tmp_35, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 1)
        tmp_35 = tmp_5 = tmp_4 = None
        tmp_37 = torch.nn.functional.relu(tmp_36, inplace=True)
        tmp_36 = None
        tmp_38 = torch.nn.functional.max_pool2d(tmp_37, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_37 = None
        tmp_39 = torch.conv2d(tmp_38, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 1)
        tmp_38 = tmp_7 = tmp_6 = None
        tmp_40 = torch.nn.functional.relu(tmp_39, inplace=True)
        tmp_39 = None
        tmp_41 = torch.conv2d(tmp_40, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_40 = tmp_9 = tmp_8 = None
        tmp_42 = torch.nn.functional.relu(tmp_41, inplace=True)
        tmp_41 = None
        tmp_43 = torch.nn.functional.max_pool2d(tmp_42, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_42 = None
        tmp_44 = torch.conv2d(tmp_43, tmp_19, tmp_18, (1, 1), (0, 0), (1, 1), 1)
        tmp_43 = tmp_19 = tmp_18 = None
        tmp_45 = torch.nn.functional.relu(tmp_44, inplace=True)
        tmp_44 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.0, False, False)
        tmp_45 = None
        tmp_47 = torch.conv2d(tmp_46, tmp_21, tmp_20, (1, 1), (0, 0), (1, 1), 1)
        tmp_46 = tmp_21 = tmp_20 = None
        tmp_48 = torch.nn.functional.relu(tmp_47, inplace=True)
        tmp_47 = None
        tmp_49 = torch.nn.functional.adaptive_avg_pool2d(tmp_48, 1)
        tmp_48 = None
        tmp_50 = tmp_49.flatten(1, -1)
        tmp_49 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.0, False, False)
        tmp_50 = None
        tmp_52 = torch.nn.functional.linear(tmp_51, tmp_17, tmp_16)
        tmp_51 = tmp_17 = tmp_16 = None
        return (tmp_52,)