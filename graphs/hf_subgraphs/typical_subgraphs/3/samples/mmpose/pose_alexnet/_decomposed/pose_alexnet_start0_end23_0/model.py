import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27):
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
        tmp_15 = in_15
        tmp_16 = in_16
        tmp_17 = in_17
        tmp_18 = in_18
        tmp_19 = in_19
        tmp_20 = in_20
        tmp_21 = in_21
        tmp_22 = in_22
        tmp_23 = in_23
        tmp_24 = in_24
        tmp_25 = in_25
        tmp_26 = in_26
        tmp_27 = in_27
        tmp_28 = torch.conv2d(tmp_0, tmp_2, tmp_1, (4, 4), (2, 2), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_29 = torch.nn.functional.relu(tmp_28, inplace=True)
        tmp_28 = None
        tmp_30 = torch.nn.functional.max_pool2d(tmp_29, 3, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_29 = None
        tmp_31 = torch.conv2d(tmp_30, tmp_6, tmp_5, (1, 1), (2, 2), (1, 1), 1)
        tmp_30 = tmp_6 = tmp_5 = None
        tmp_32 = torch.nn.functional.relu(tmp_31, inplace=True)
        tmp_31 = None
        tmp_33 = torch.nn.functional.max_pool2d(tmp_32, 3, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_32 = None
        tmp_34 = torch.conv2d(tmp_33, tmp_8, tmp_7, (1, 1), (1, 1), (1, 1), 1)
        tmp_33 = tmp_8 = tmp_7 = None
        tmp_35 = torch.nn.functional.relu(tmp_34, inplace=True)
        tmp_34 = None
        tmp_36 = torch.conv2d(tmp_35, tmp_10, tmp_9, (1, 1), (1, 1), (1, 1), 1)
        tmp_35 = tmp_10 = tmp_9 = None
        tmp_37 = torch.nn.functional.relu(tmp_36, inplace=True)
        tmp_36 = None
        tmp_38 = torch.conv2d(tmp_37, tmp_4, tmp_3, (1, 1), (1, 1), (1, 1), 1)
        tmp_37 = tmp_4 = tmp_3 = None
        tmp_39 = torch.nn.functional.relu(tmp_38, inplace=True)
        tmp_38 = None
        tmp_40 = torch.nn.functional.max_pool2d(tmp_39, 3, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_39 = None
        tmp_41 = torch.conv_transpose2d(tmp_40, tmp_11, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_40 = tmp_11 = None
        tmp_42 = torch.nn.functional.batch_norm(tmp_41, tmp_12, tmp_13, tmp_15, tmp_14, False, 0.1, 1e-05)
        tmp_41 = tmp_12 = tmp_13 = tmp_15 = tmp_14 = None
        tmp_43 = torch.nn.functional.relu(tmp_42, inplace=True)
        tmp_42 = None
        tmp_44 = torch.conv_transpose2d(tmp_43, tmp_16, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_43 = tmp_16 = None
        tmp_45 = torch.nn.functional.batch_norm(tmp_44, tmp_17, tmp_18, tmp_20, tmp_19, False, 0.1, 1e-05)
        tmp_44 = tmp_17 = tmp_18 = tmp_20 = tmp_19 = None
        tmp_46 = torch.nn.functional.relu(tmp_45, inplace=True)
        tmp_45 = None
        tmp_47 = torch.conv_transpose2d(tmp_46, tmp_21, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_46 = tmp_21 = None
        tmp_48 = torch.nn.functional.batch_norm(tmp_47, tmp_22, tmp_23, tmp_25, tmp_24, False, 0.1, 1e-05)
        tmp_47 = tmp_22 = tmp_23 = tmp_25 = tmp_24 = None
        tmp_49 = torch.nn.functional.relu(tmp_48, inplace=True)
        tmp_48 = None
        tmp_50 = torch.conv2d(tmp_49, tmp_27, tmp_26, (1, 1), (0, 0), (1, 1), 1)
        tmp_49 = tmp_27 = tmp_26 = None
        return (tmp_50,)