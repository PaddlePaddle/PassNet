import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_9 = torch.nn.functional.linear(in_1, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_10 = torch.nn.functional.linear(in_2, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_11 = in_3.view(1, 300, 8, 32)
        tmp_12 = torch.nn.functional.linear(in_4, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_13 = tmp_12.view(1, 300, 8, 32)
        tmp_12 = None
        tmp_14 = torch.cat([tmp_11, tmp_13], dim=3)
        tmp_11 = tmp_13 = None
        tmp_15 = tmp_14.view(1, 300, 512)
        tmp_14 = None
        tmp_16 = tmp_8.view(1, 625, 8, 32)
        tmp_8 = None
        tmp_17 = tmp_10.view(1, 625, 8, 32)
        tmp_10 = None
        tmp_18 = torch.cat([tmp_16, tmp_17], dim=3)
        tmp_16 = tmp_17 = None
        tmp_19 = tmp_18.view(1, 625, 512)
        tmp_18 = None
        tmp_20 = tmp_15 * 0.125
        tmp_15 = None
        tmp_21 = tmp_19.view(1, -1, 8, 64)
        tmp_19 = None
        tmp_22 = tmp_21.transpose(1, 2)
        tmp_21 = None
        tmp_23 = tmp_22.contiguous()
        tmp_22 = None
        tmp_24 = tmp_9.view(1, -1, 8, 32)
        tmp_9 = None
        tmp_25 = tmp_24.transpose(1, 2)
        tmp_24 = None
        tmp_26 = tmp_25.contiguous()
        tmp_25 = None
        tmp_27 = tmp_20.view(1, 300, 8, 64)
        tmp_20 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = tmp_28.contiguous()
        tmp_28 = None
        tmp_30 = tmp_29.view(8, -1, 64)
        tmp_29 = None
        tmp_31 = tmp_23.view(8, -1, 64)
        tmp_23 = None
        tmp_32 = tmp_26.view(8, -1, 32)
        tmp_26 = None
        tmp_33 = tmp_31.transpose(1, 2)
        tmp_31 = None
        tmp_34 = torch.bmm(tmp_30, tmp_33)
        tmp_30 = tmp_33 = None
        tmp_35 = tmp_34.view(1, 8, 300, 625)
        tmp_34 = None
        tmp_36 = tmp_35 + in_0
        tmp_35 = None
        tmp_37 = tmp_36.view(8, 300, 625)
        tmp_36 = None
        tmp_38 = torch.nn.functional.softmax(tmp_37, dim=-1)
        tmp_37 = None
        tmp_39 = tmp_38.view(1, 8, 300, 625)
        tmp_38 = None
        tmp_40 = tmp_39.view(8, 300, 625)
        tmp_41 = torch.nn.functional.dropout(tmp_40, p=0.0, training=False)
        tmp_40 = None
        tmp_42 = torch.bmm(tmp_41, tmp_32)
        tmp_41 = tmp_32 = None
        tmp_43 = tmp_42.view(1, 8, 300, 32)
        tmp_42 = None
        tmp_44 = tmp_43.transpose(1, 2)
        tmp_43 = None
        tmp_45 = tmp_44.reshape(1, 300, 256)
        tmp_44 = None
        return (tmp_45, tmp_39)