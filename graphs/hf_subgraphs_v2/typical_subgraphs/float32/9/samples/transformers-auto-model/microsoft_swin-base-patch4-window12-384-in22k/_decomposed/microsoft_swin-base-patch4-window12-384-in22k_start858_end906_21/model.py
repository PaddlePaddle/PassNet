import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1, in_2, in_3, in_4):
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
        tmp_14 = torch.nn.functional.linear(in_1, tmp_6, tmp_5)
        tmp_6 = tmp_5 = None
        tmp_15 = tmp_14.view((4, 144, -1, 32))
        tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = in_2.transpose(-1, -2)
        tmp_18 = torch.matmul(in_4, tmp_17)
        tmp_17 = None
        tmp_19 = tmp_18 / 5.656854249492381
        tmp_18 = None
        tmp_20 = tmp_4.view(-1)
        tmp_4 = None
        tmp_21 = tmp_7[tmp_20]
        tmp_7 = tmp_20 = None
        tmp_22 = tmp_21.view(144, 144, -1)
        tmp_21 = None
        tmp_23 = tmp_22.permute(2, 0, 1)
        tmp_22 = None
        tmp_24 = tmp_23.contiguous()
        tmp_23 = None
        tmp_25 = tmp_24.unsqueeze(0)
        tmp_24 = None
        tmp_26 = tmp_19 + tmp_25
        tmp_19 = tmp_25 = None
        tmp_27 = tmp_26.view(1, 4, 16, 144, 144)
        tmp_26 = None
        tmp_28 = in_0.unsqueeze(1)
        tmp_29 = tmp_28.unsqueeze(0)
        tmp_28 = None
        tmp_30 = tmp_27 + tmp_29
        tmp_27 = tmp_29 = None
        tmp_31 = tmp_30.view(-1, 16, 144, 144)
        tmp_30 = None
        tmp_32 = torch.nn.functional.softmax(tmp_31, dim=-1)
        tmp_31 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.0, False, False)
        tmp_32 = None
        tmp_34 = torch.matmul(tmp_33, tmp_16)
        tmp_33 = tmp_16 = None
        tmp_35 = tmp_34.permute(0, 2, 1, 3)
        tmp_34 = None
        tmp_36 = tmp_35.contiguous()
        tmp_35 = None
        tmp_37 = tmp_36.view((4, 144, 512))
        tmp_36 = None
        tmp_38 = torch.nn.functional.linear(tmp_37, tmp_3, tmp_2)
        tmp_37 = tmp_3 = tmp_2 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, 0.0, False, False)
        tmp_38 = None
        tmp_40 = tmp_39.view(-1, 12, 12, 512)
        tmp_39 = None
        tmp_41 = tmp_40.view(-1, 2, 2, 12, 12, 512)
        tmp_40 = None
        tmp_42 = tmp_41.permute(0, 1, 3, 2, 4, 5)
        tmp_41 = None
        tmp_43 = tmp_42.contiguous()
        tmp_42 = None
        tmp_44 = tmp_43.view(-1, 24, 24, 512)
        tmp_43 = None
        tmp_45 = torch.roll(tmp_44, shifts=(6, 6), dims=(1, 2))
        tmp_44 = None
        tmp_46 = tmp_45.view(1, 576, 512)
        tmp_45 = None
        tmp_47 = in_3 + tmp_46
        tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_47, (512,), tmp_11, tmp_10, 1e-05)
        tmp_11 = tmp_10 = None
        tmp_49 = torch.nn.functional.linear(tmp_48, tmp_9, tmp_8)
        tmp_48 = tmp_9 = tmp_8 = None
        tmp_50 = torch.nn.functional.gelu(tmp_49)
        tmp_49 = None
        tmp_51 = torch.nn.functional.linear(tmp_50, tmp_13, tmp_12)
        tmp_50 = tmp_13 = tmp_12 = None
        tmp_52 = torch.nn.functional.dropout(tmp_51, 0.0, False, False)
        tmp_51 = None
        tmp_53 = tmp_47 + tmp_52
        tmp_47 = tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (512,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_55 = tmp_54.view(1, 24, 24, 512)
        tmp_54 = None
        tmp_56 = torch.nn.functional.pad(tmp_55, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_55 = None
        tmp_57 = tmp_56.view(1, 2, 12, 2, 12, 512)
        tmp_56 = None
        tmp_58 = tmp_57.permute(0, 1, 3, 2, 4, 5)
        tmp_57 = None
        tmp_59 = tmp_58.contiguous()
        tmp_58 = None
        tmp_60 = tmp_59.view(-1, 12, 12, 512)
        tmp_59 = None
        tmp_61 = tmp_60.view(-1, 144, 512)
        tmp_60 = None
        return (tmp_61, tmp_53)