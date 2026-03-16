import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, in_0, in_1, in_2, in_3):
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
        tmp_20 = torch.nn.functional.linear(in_0, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_21 = tmp_20.view(4, -1, 4, 16)
        tmp_20 = None
        tmp_22 = tmp_21.transpose(1, 2)
        tmp_21 = None
        tmp_23 = torch.nn.functional.normalize(in_3, dim=-1)
        tmp_24 = torch.nn.functional.normalize(in_2, dim=-1)
        tmp_25 = tmp_24.transpose(-2, -1)
        tmp_24 = None
        tmp_26 = tmp_23 @ tmp_25
        tmp_23 = tmp_25 = None
        tmp_27 = torch.clamp(tmp_9, max=4.605170185988092)
        tmp_9 = None
        tmp_28 = tmp_27.exp()
        tmp_27 = None
        tmp_29 = tmp_26 * tmp_28
        tmp_26 = tmp_28 = None
        tmp_30 = torch.nn.functional.linear(tmp_2, tmp_5, tmp_4)
        tmp_2 = tmp_5 = tmp_4 = None
        tmp_31 = torch.nn.functional.relu(tmp_30, inplace=True)
        tmp_30 = None
        tmp_32 = torch.nn.functional.linear(tmp_31, tmp_6, None)
        tmp_31 = tmp_6 = None
        tmp_33 = tmp_32.view(-1, 4)
        tmp_32 = None
        tmp_34 = tmp_3.view(-1)
        tmp_3 = None
        tmp_35 = tmp_33[tmp_34]
        tmp_33 = tmp_34 = None
        tmp_36 = tmp_35.view(4, 4, -1)
        tmp_35 = None
        tmp_37 = tmp_36.permute(2, 0, 1)
        tmp_36 = None
        tmp_38 = tmp_37.contiguous()
        tmp_37 = None
        tmp_39 = torch.sigmoid(tmp_38)
        tmp_38 = None
        tmp_40 = 16 * tmp_39
        tmp_39 = None
        tmp_41 = tmp_40.unsqueeze(0)
        tmp_40 = None
        tmp_42 = tmp_29 + tmp_41
        tmp_29 = tmp_41 = None
        tmp_43 = torch.nn.functional.softmax(tmp_42, dim=-1)
        tmp_42 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.0, False, False)
        tmp_43 = None
        tmp_45 = torch.matmul(tmp_44, tmp_22)
        tmp_44 = tmp_22 = None
        tmp_46 = tmp_45.permute(0, 2, 1, 3)
        tmp_45 = None
        tmp_47 = tmp_46.contiguous()
        tmp_46 = None
        tmp_48 = tmp_47.view((4, 4, 64))
        tmp_47 = None
        tmp_49 = torch.nn.functional.linear(tmp_48, tmp_1, tmp_0)
        tmp_48 = tmp_1 = tmp_0 = None
        tmp_50 = torch.nn.functional.dropout(tmp_49, 0.0, False, False)
        tmp_49 = None
        tmp_51 = tmp_50.view(-1, 2, 2, 64)
        tmp_50 = None
        tmp_52 = tmp_51.view(-1, 2, 2, 2, 2, 64)
        tmp_51 = None
        tmp_53 = tmp_52.permute(0, 1, 3, 2, 4, 5)
        tmp_52 = None
        tmp_54 = tmp_53.contiguous()
        tmp_53 = None
        tmp_55 = tmp_54.view(-1, 4, 4, 64)
        tmp_54 = None
        tmp_56 = tmp_55.view(1, 16, 64)
        tmp_55 = None
        tmp_57 = torch.nn.functional.layer_norm(tmp_56, (64,), tmp_15, tmp_14, 1e-05)
        tmp_56 = tmp_15 = tmp_14 = None
        tmp_58 = in_1 + tmp_57
        tmp_57 = None
        tmp_59 = torch.nn.functional.linear(tmp_58, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_60 = torch.nn.functional.gelu(tmp_59)
        tmp_59 = None
        tmp_61 = torch.nn.functional.linear(tmp_60, tmp_17, tmp_16)
        tmp_60 = tmp_17 = tmp_16 = None
        tmp_62 = torch.nn.functional.dropout(tmp_61, 0.0, False, False)
        tmp_61 = None
        tmp_63 = torch.nn.functional.layer_norm(tmp_62, (64,), tmp_13, tmp_12, 1e-05)
        tmp_62 = tmp_13 = tmp_12 = None
        tmp_64 = tmp_58 + tmp_63
        tmp_58 = tmp_63 = None
        tmp_65 = torch.nn.functional.layer_norm(tmp_64, (64,), tmp_19, tmp_18, 1e-05)
        tmp_64 = tmp_19 = tmp_18 = None
        tmp_66 = tmp_65.transpose(1, 2)
        tmp_67 = torch.adaptive_avg_pool1d(tmp_66, 1)
        tmp_66 = None
        tmp_68 = torch.flatten(tmp_67, 1)
        tmp_67 = None
        return (tmp_65, tmp_68)