import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
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
        tmp_14 = torch.nn.functional.linear(in_14, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_15 = tmp_14.view((1, 144, -1, 32))
        tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = in_16.transpose(-1, -2)
        tmp_18 = torch.matmul(in_17, tmp_17)
        tmp_17 = None
        tmp_19 = tmp_18 / 5.656854249492381
        tmp_18 = None
        tmp_20 = tmp_2.view(-1)
        tmp_2 = None
        tmp_21 = tmp_5[tmp_20]
        tmp_5 = tmp_20 = None
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
        tmp_27 = torch.nn.functional.softmax(tmp_26, dim=-1)
        tmp_26 = None
        tmp_28 = torch.nn.functional.dropout(tmp_27, 0.0, False, False)
        tmp_27 = None
        tmp_29 = torch.matmul(tmp_28, tmp_16)
        tmp_28 = tmp_16 = None
        tmp_30 = tmp_29.permute(0, 2, 1, 3)
        tmp_29 = None
        tmp_31 = tmp_30.contiguous()
        tmp_30 = None
        tmp_32 = tmp_31.view((1, 144, 1024))
        tmp_31 = None
        tmp_33 = torch.nn.functional.linear(tmp_32, tmp_1, tmp_0)
        tmp_32 = tmp_1 = tmp_0 = None
        tmp_34 = torch.nn.functional.dropout(tmp_33, 0.0, False, False)
        tmp_33 = None
        tmp_35 = tmp_34.view(-1, 12, 12, 1024)
        tmp_34 = None
        tmp_36 = tmp_35.view(-1, 1, 1, 12, 12, 1024)
        tmp_35 = None
        tmp_37 = tmp_36.permute(0, 1, 3, 2, 4, 5)
        tmp_36 = None
        tmp_38 = tmp_37.contiguous()
        tmp_37 = None
        tmp_39 = tmp_38.view(-1, 12, 12, 1024)
        tmp_38 = None
        tmp_40 = tmp_39.view(1, 144, 1024)
        tmp_39 = None
        tmp_41 = in_15 + tmp_40
        tmp_40 = None
        tmp_42 = torch.nn.functional.layer_norm(tmp_41, (1024,), tmp_9, tmp_8, 1e-05)
        tmp_9 = tmp_8 = None
        tmp_43 = torch.nn.functional.linear(tmp_42, tmp_7, tmp_6)
        tmp_42 = tmp_7 = tmp_6 = None
        tmp_44 = torch.nn.functional.gelu(tmp_43)
        tmp_43 = None
        tmp_45 = torch.nn.functional.linear(tmp_44, tmp_11, tmp_10)
        tmp_44 = tmp_11 = tmp_10 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.0, False, False)
        tmp_45 = None
        tmp_47 = tmp_41 + tmp_46
        tmp_41 = tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_47, (1024,), tmp_13, tmp_12, 1e-05)
        tmp_13 = tmp_12 = None
        tmp_49 = tmp_48.view(1, 12, 12, 1024)
        tmp_48 = None
        tmp_50 = torch.nn.functional.pad(tmp_49, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_49 = None
        tmp_51 = tmp_50.view(1, 1, 12, 1, 12, 1024)
        tmp_50 = None
        tmp_52 = tmp_51.permute(0, 1, 3, 2, 4, 5)
        tmp_51 = None
        tmp_53 = tmp_52.contiguous()
        tmp_52 = None
        tmp_54 = tmp_53.view(-1, 12, 12, 1024)
        tmp_53 = None
        tmp_55 = tmp_54.view(-1, 144, 1024)
        tmp_54 = None
        return (tmp_55, tmp_47)