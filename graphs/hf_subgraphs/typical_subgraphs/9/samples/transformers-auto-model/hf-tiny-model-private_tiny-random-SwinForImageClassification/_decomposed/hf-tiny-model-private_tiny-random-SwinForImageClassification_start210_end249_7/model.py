import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0, in_1, in_2, in_3):
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
        tmp_16 = torch.nn.functional.linear(in_0, tmp_6, tmp_5)
        tmp_6 = tmp_5 = None
        tmp_17 = tmp_16.view((4, 4, -1, 16))
        tmp_16 = None
        tmp_18 = tmp_17.transpose(1, 2)
        tmp_17 = None
        tmp_19 = in_2.transpose(-1, -2)
        tmp_20 = torch.matmul(in_3, tmp_19)
        tmp_19 = None
        tmp_21 = tmp_20 / 4.0
        tmp_20 = None
        tmp_22 = tmp_4.view(-1)
        tmp_4 = None
        tmp_23 = tmp_7[tmp_22]
        tmp_7 = tmp_22 = None
        tmp_24 = tmp_23.view(4, 4, -1)
        tmp_23 = None
        tmp_25 = tmp_24.permute(2, 0, 1)
        tmp_24 = None
        tmp_26 = tmp_25.contiguous()
        tmp_25 = None
        tmp_27 = tmp_26.unsqueeze(0)
        tmp_26 = None
        tmp_28 = tmp_21 + tmp_27
        tmp_21 = tmp_27 = None
        tmp_29 = torch.nn.functional.softmax(tmp_28, dim=-1)
        tmp_28 = None
        tmp_30 = torch.nn.functional.dropout(tmp_29, 0.0, False, False)
        tmp_29 = None
        tmp_31 = torch.matmul(tmp_30, tmp_18)
        tmp_30 = tmp_18 = None
        tmp_32 = tmp_31.permute(0, 2, 1, 3)
        tmp_31 = None
        tmp_33 = tmp_32.contiguous()
        tmp_32 = None
        tmp_34 = tmp_33.view((4, 4, 64))
        tmp_33 = None
        tmp_35 = torch.nn.functional.linear(tmp_34, tmp_3, tmp_2)
        tmp_34 = tmp_3 = tmp_2 = None
        tmp_36 = torch.nn.functional.dropout(tmp_35, 0.0, False, False)
        tmp_35 = None
        tmp_37 = tmp_36.view(-1, 2, 2, 64)
        tmp_36 = None
        tmp_38 = tmp_37.view(-1, 2, 2, 2, 2, 64)
        tmp_37 = None
        tmp_39 = tmp_38.permute(0, 1, 3, 2, 4, 5)
        tmp_38 = None
        tmp_40 = tmp_39.contiguous()
        tmp_39 = None
        tmp_41 = tmp_40.view(-1, 4, 4, 64)
        tmp_40 = None
        tmp_42 = tmp_41.view(1, 16, 64)
        tmp_41 = None
        tmp_43 = in_1 + tmp_42
        tmp_42 = None
        tmp_44 = torch.nn.functional.layer_norm(tmp_43, (64,), tmp_11, tmp_10, 1e-05)
        tmp_11 = tmp_10 = None
        tmp_45 = torch.nn.functional.linear(tmp_44, tmp_9, tmp_8)
        tmp_44 = tmp_9 = tmp_8 = None
        tmp_46 = torch.nn.functional.gelu(tmp_45)
        tmp_45 = None
        tmp_47 = torch.nn.functional.linear(tmp_46, tmp_13, tmp_12)
        tmp_46 = tmp_13 = tmp_12 = None
        tmp_48 = torch.nn.functional.dropout(tmp_47, 0.0, False, False)
        tmp_47 = None
        tmp_49 = tmp_43 + tmp_48
        tmp_43 = tmp_48 = None
        tmp_50 = torch.nn.functional.layer_norm(tmp_49, (64,), tmp_15, tmp_14, 1e-05)
        tmp_49 = tmp_15 = tmp_14 = None
        tmp_51 = tmp_50.transpose(1, 2)
        tmp_50 = None
        tmp_52 = torch.adaptive_avg_pool1d(tmp_51, 1)
        tmp_51 = None
        tmp_53 = torch.flatten(tmp_52, 1)
        tmp_52 = None
        tmp_54 = torch.nn.functional.linear(tmp_53, tmp_1, tmp_0)
        tmp_53 = tmp_1 = tmp_0 = None
        return (tmp_54,)