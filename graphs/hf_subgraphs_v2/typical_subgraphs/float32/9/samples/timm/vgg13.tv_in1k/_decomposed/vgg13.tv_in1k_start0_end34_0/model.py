import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, in_0):
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
        tmp_22 = w_22
        tmp_23 = w_23
        tmp_24 = w_24
        tmp_25 = w_25
        tmp_26 = in_0
        tmp_27 = torch.conv2d(tmp_26, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_26 = tmp_1 = tmp_0 = None
        tmp_28 = torch.nn.functional.relu(tmp_27, inplace=True)
        tmp_27 = None
        tmp_29 = torch.conv2d(tmp_28, tmp_15, tmp_14, (1, 1), (1, 1), (1, 1), 1)
        tmp_28 = tmp_15 = tmp_14 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace=True)
        tmp_29 = None
        tmp_31 = torch.nn.functional.max_pool2d(tmp_30, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_30 = None
        tmp_32 = torch.conv2d(tmp_31, tmp_17, tmp_16, (1, 1), (1, 1), (1, 1), 1)
        tmp_31 = tmp_17 = tmp_16 = None
        tmp_33 = torch.nn.functional.relu(tmp_32, inplace=True)
        tmp_32 = None
        tmp_34 = torch.conv2d(tmp_33, tmp_19, tmp_18, (1, 1), (1, 1), (1, 1), 1)
        tmp_33 = tmp_19 = tmp_18 = None
        tmp_35 = torch.nn.functional.relu(tmp_34, inplace=True)
        tmp_34 = None
        tmp_36 = torch.nn.functional.max_pool2d(tmp_35, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_35 = None
        tmp_37 = torch.conv2d(tmp_36, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1)
        tmp_36 = tmp_3 = tmp_2 = None
        tmp_38 = torch.nn.functional.relu(tmp_37, inplace=True)
        tmp_37 = None
        tmp_39 = torch.conv2d(tmp_38, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 1)
        tmp_38 = tmp_5 = tmp_4 = None
        tmp_40 = torch.nn.functional.relu(tmp_39, inplace=True)
        tmp_39 = None
        tmp_41 = torch.nn.functional.max_pool2d(tmp_40, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_40 = None
        tmp_42 = torch.conv2d(tmp_41, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 1)
        tmp_41 = tmp_7 = tmp_6 = None
        tmp_43 = torch.nn.functional.relu(tmp_42, inplace=True)
        tmp_42 = None
        tmp_44 = torch.conv2d(tmp_43, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_43 = tmp_9 = tmp_8 = None
        tmp_45 = torch.nn.functional.relu(tmp_44, inplace=True)
        tmp_44 = None
        tmp_46 = torch.nn.functional.max_pool2d(tmp_45, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_45 = None
        tmp_47 = torch.conv2d(tmp_46, tmp_11, tmp_10, (1, 1), (1, 1), (1, 1), 1)
        tmp_46 = tmp_11 = tmp_10 = None
        tmp_48 = torch.nn.functional.relu(tmp_47, inplace=True)
        tmp_47 = None
        tmp_49 = torch.conv2d(tmp_48, tmp_13, tmp_12, (1, 1), (1, 1), (1, 1), 1)
        tmp_48 = tmp_13 = tmp_12 = None
        tmp_50 = torch.nn.functional.relu(tmp_49, inplace=True)
        tmp_49 = None
        tmp_51 = torch.nn.functional.max_pool2d(tmp_50, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_50 = None
        tmp_52 = torch.conv2d(tmp_51, tmp_23, tmp_22, (1, 1), (0, 0), (1, 1), 1)
        tmp_51 = tmp_23 = tmp_22 = None
        tmp_53 = torch.nn.functional.relu(tmp_52, inplace=True)
        tmp_52 = None
        tmp_54 = torch.nn.functional.dropout(tmp_53, 0.0, False, False)
        tmp_53 = None
        tmp_55 = torch.conv2d(tmp_54, tmp_25, tmp_24, (1, 1), (0, 0), (1, 1), 1)
        tmp_54 = tmp_25 = tmp_24 = None
        tmp_56 = torch.nn.functional.relu(tmp_55, inplace=True)
        tmp_55 = None
        tmp_57 = torch.nn.functional.adaptive_avg_pool2d(tmp_56, 1)
        tmp_56 = None
        tmp_58 = tmp_57.flatten(1, -1)
        tmp_57 = None
        tmp_59 = torch.nn.functional.dropout(tmp_58, 0.0, False, False)
        tmp_58 = None
        tmp_60 = torch.nn.functional.linear(tmp_59, tmp_21, tmp_20)
        tmp_59 = tmp_21 = tmp_20 = None
        return (tmp_60,)