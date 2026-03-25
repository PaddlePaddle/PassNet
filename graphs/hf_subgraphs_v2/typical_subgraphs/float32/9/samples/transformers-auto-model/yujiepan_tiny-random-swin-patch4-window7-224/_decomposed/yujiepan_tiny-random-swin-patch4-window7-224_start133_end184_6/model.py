import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, in_0, in_1, in_2, in_3):
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
        tmp_17 = torch.nn.functional.linear(in_0, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_18 = tmp_17.view((4, 49, -1, 4))
        tmp_17 = None
        tmp_19 = tmp_18.transpose(1, 2)
        tmp_18 = None
        tmp_20 = in_2.transpose(-1, -2)
        tmp_21 = torch.matmul(in_3, tmp_20)
        tmp_20 = None
        tmp_22 = tmp_21 / 2.0
        tmp_21 = None
        tmp_23 = tmp_2.view(-1)
        tmp_2 = None
        tmp_24 = tmp_5[tmp_23]
        tmp_5 = tmp_23 = None
        tmp_25 = tmp_24.view(49, 49, -1)
        tmp_24 = None
        tmp_26 = tmp_25.permute(2, 0, 1)
        tmp_25 = None
        tmp_27 = tmp_26.contiguous()
        tmp_26 = None
        tmp_28 = tmp_27.unsqueeze(0)
        tmp_27 = None
        tmp_29 = tmp_22 + tmp_28
        tmp_22 = tmp_28 = None
        tmp_30 = torch.nn.functional.softmax(tmp_29, dim=-1)
        tmp_29 = None
        tmp_31 = torch.nn.functional.dropout(tmp_30, 0.0, False, False)
        tmp_30 = None
        tmp_32 = torch.matmul(tmp_31, tmp_19)
        tmp_31 = tmp_19 = None
        tmp_33 = tmp_32.permute(0, 2, 1, 3)
        tmp_32 = None
        tmp_34 = tmp_33.contiguous()
        tmp_33 = None
        tmp_35 = tmp_34.view((4, 49, 32))
        tmp_34 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_1, tmp_0)
        tmp_35 = tmp_1 = tmp_0 = None
        tmp_37 = torch.nn.functional.dropout(tmp_36, 0.0, False, False)
        tmp_36 = None
        tmp_38 = tmp_37.view(-1, 7, 7, 32)
        tmp_37 = None
        tmp_39 = tmp_38.view(-1, 2, 2, 7, 7, 32)
        tmp_38 = None
        tmp_40 = tmp_39.permute(0, 1, 3, 2, 4, 5)
        tmp_39 = None
        tmp_41 = tmp_40.contiguous()
        tmp_40 = None
        tmp_42 = tmp_41.view(-1, 14, 14, 32)
        tmp_41 = None
        tmp_43 = tmp_42.view(1, 196, 32)
        tmp_42 = None
        tmp_44 = in_1 + tmp_43
        tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (32,), tmp_9, tmp_8, 1e-05)
        tmp_9 = tmp_8 = None
        tmp_46 = torch.nn.functional.linear(tmp_45, tmp_7, tmp_6)
        tmp_45 = tmp_7 = tmp_6 = None
        tmp_47 = torch.nn.functional.gelu(tmp_46)
        tmp_46 = None
        tmp_48 = torch.nn.functional.linear(tmp_47, tmp_11, tmp_10)
        tmp_47 = tmp_11 = tmp_10 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.0, False, False)
        tmp_48 = None
        tmp_50 = tmp_44 + tmp_49
        tmp_44 = tmp_49 = None
        tmp_51 = tmp_50.view(1, 14, 14, 32)
        tmp_50 = None
        tmp_52 = tmp_51[slice(None, None, None), slice(0, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_53 = tmp_51[slice(None, None, None), slice(1, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_54 = tmp_51[slice(None, None, None), slice(0, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_55 = tmp_51[slice(None, None, None), slice(1, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_51 = None
        tmp_56 = torch.cat([tmp_52, tmp_53, tmp_54, tmp_55], -1)
        tmp_52 = tmp_53 = tmp_54 = tmp_55 = None
        tmp_57 = tmp_56.view(1, -1, 128)
        tmp_56 = None
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (128,), tmp_13, tmp_12, 1e-05)
        tmp_57 = tmp_13 = tmp_12 = None
        tmp_59 = torch.nn.functional.linear(tmp_58, tmp_14, None)
        tmp_58 = tmp_14 = None
        tmp_60 = torch.nn.functional.layer_norm(tmp_59, (64,), tmp_16, tmp_15, 1e-05)
        tmp_16 = tmp_15 = None
        tmp_61 = tmp_60.view(1, 7, 7, 64)
        tmp_60 = None
        tmp_62 = torch.nn.functional.pad(tmp_61, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_61 = None
        tmp_63 = tmp_62.view(1, 1, 7, 1, 7, 64)
        tmp_62 = None
        tmp_64 = tmp_63.permute(0, 1, 3, 2, 4, 5)
        tmp_63 = None
        tmp_65 = tmp_64.contiguous()
        tmp_64 = None
        tmp_66 = tmp_65.view(-1, 7, 7, 64)
        tmp_65 = None
        tmp_67 = tmp_66.view(-1, 49, 64)
        tmp_66 = None
        return (tmp_67, tmp_59)