import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0, in_1, in_2):
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
        tmp_11 = torch.nn.functional.linear(in_1, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_12 = in_1.transpose(1, 2)
        tmp_13 = torch.conv1d(tmp_12, tmp_4, None, (1,), (4,), (1,), 32)
        tmp_12 = tmp_4 = None
        tmp_14 = torch.conv1d(tmp_13, tmp_5, None, (1,), (0,), (1,), 1)
        tmp_13 = tmp_5 = None
        tmp_14 += tmp_6
        tmp_15 = tmp_14
        tmp_14 = tmp_6 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(in_1, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_18 = tmp_17.view(1, -1, 2, 8)
        tmp_19 = tmp_18.transpose(1, 2)
        tmp_18 = None
        tmp_20 = in_2.view(1, -1, 2, 8)
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = tmp_11.view(1, -1, 2, 8)
        tmp_11 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = torch.multiply(tmp_16, tmp_17)
        tmp_16 = tmp_17 = None
        tmp_25 = torch.nn.functional.linear(tmp_24, tmp_1, tmp_0)
        tmp_24 = tmp_1 = tmp_0 = None
        tmp_26 = torch.reshape(tmp_25, [-1, 9, 1])
        tmp_25 = None
        tmp_27 = torch.softmax(tmp_26, dim=1)
        tmp_26 = None
        tmp_28 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_29 = torch.reshape(tmp_28, [1, -1, 16])
        tmp_28 = None
        tmp_30 = tmp_29.transpose(1, 2)
        tmp_29 = None
        tmp_31 = tmp_30.contiguous()
        tmp_30 = None
        tmp_32 = tmp_31.unsqueeze(-1)
        tmp_31 = None
        tmp_33 = torch.nn.functional.unfold(tmp_32, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
        tmp_32 = None
        tmp_34 = tmp_33.transpose(1, 2)
        tmp_33 = None
        tmp_35 = tmp_34.reshape(1, -1, 16, 9)
        tmp_34 = None
        tmp_36 = torch.reshape(tmp_35, [-1, 8, 9])
        tmp_35 = None
        tmp_37 = torch.matmul(tmp_36, tmp_27)
        tmp_36 = tmp_27 = None
        tmp_38 = torch.reshape(tmp_37, [-1, 16])
        tmp_37 = None
        tmp_39 = tmp_21.transpose(-1, -2)
        tmp_21 = None
        tmp_40 = torch.matmul(tmp_19, tmp_39)
        tmp_19 = tmp_39 = None
        tmp_41 = tmp_40 / 2.8284271247461903
        tmp_40 = None
        tmp_42 = tmp_41 + in_0
        tmp_41 = None
        tmp_43 = torch.nn.functional.softmax(tmp_42, dim=-1)
        tmp_42 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.1, False, False)
        tmp_43 = None
        tmp_45 = torch.matmul(tmp_44, tmp_23)
        tmp_44 = tmp_23 = None
        tmp_46 = tmp_45.permute(0, 2, 1, 3)
        tmp_45 = None
        tmp_47 = tmp_46.contiguous()
        tmp_46 = None
        tmp_48 = torch.reshape(tmp_38, [1, -1, 2, 8])
        tmp_38 = None
        tmp_49 = torch.cat([tmp_47, tmp_48], 2)
        tmp_47 = tmp_48 = None
        tmp_50 = tmp_49.view(1, 45, 32)
        tmp_49 = None
        return (tmp_50,)