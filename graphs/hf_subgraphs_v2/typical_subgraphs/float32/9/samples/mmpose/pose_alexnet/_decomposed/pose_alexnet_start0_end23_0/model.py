import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = w_10
        tmp_12 = w_11
        tmp_13 = w_12
        tmp_14 = w_13
        tmp_15 = w_14
        tmp_16 = w_15
        tmp_17 = w_16
        tmp_18 = w_17
        tmp_19 = w_18
        tmp_20 = w_19
        tmp_21 = w_20
        tmp_22 = w_21
        tmp_23 = w_22
        tmp_24 = w_23
        tmp_25 = w_24
        tmp_26 = w_25
        tmp_27 = w_26
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