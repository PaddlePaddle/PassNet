import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20):
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
        in_20 += in_19
        tmp_19 = in_20
        tmp_20 = torch.nn.functional.relu(tmp_19, inplace=True)
        tmp_19 = None
        tmp_21 = torch.conv_transpose2d(tmp_20, tmp_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_20 = tmp_0 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_21, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_21 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.conv_transpose2d(tmp_23, tmp_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_23 = tmp_5 = None
        tmp_25 = torch.nn.functional.batch_norm(tmp_24, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 1e-05)
        tmp_24 = tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace=True)
        tmp_25 = None
        tmp_27 = torch.conv_transpose2d(tmp_26, tmp_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_26 = tmp_10 = None
        tmp_28 = torch.nn.functional.batch_norm(tmp_27, tmp_11, tmp_12, tmp_14, tmp_13, False, 0.1, 1e-05)
        tmp_27 = tmp_11 = tmp_12 = tmp_14 = tmp_13 = None
        tmp_29 = torch.nn.functional.relu(tmp_28, inplace=True)
        tmp_28 = None
        tmp_30 = torch.conv2d(tmp_29, tmp_16, tmp_15, (1, 1), (0, 0), (1, 1), 1)
        tmp_29 = tmp_16 = tmp_15 = None
        tmp_31 = tmp_30 * 1.0
        tmp_30 = None
        tmp_32 = tmp_31.reshape(-1, 17, 4096)
        tmp_31 = None
        tmp_33 = torch.nn.functional.softmax(tmp_32, dim=2)
        tmp_32 = None
        tmp_34 = tmp_33.reshape(-1, 17, 64, 64)
        tmp_33 = None
        tmp_35 = tmp_34.mul(tmp_17)
        tmp_17 = None
        tmp_36 = tmp_35.reshape(256, 17, -1)
        tmp_35 = None
        tmp_37 = torch.sum(tmp_36, dim=2, keepdim=True)
        tmp_36 = None
        tmp_38 = tmp_34.mul(tmp_18)
        tmp_18 = None
        tmp_39 = tmp_38.reshape(256, 17, -1)
        tmp_38 = None
        tmp_40 = torch.sum(tmp_39, dim=2, keepdim=True)
        tmp_39 = None
        tmp_41 = torch.cat([tmp_37, tmp_40], dim=-1)
        tmp_37 = tmp_40 = None
        return (tmp_34, tmp_41)