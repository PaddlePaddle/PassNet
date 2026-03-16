import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, in_0, in_1, in_2, in_3):
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
        tmp_20 = w_20
        tmp_21 = torch.nn.functional.linear(in_1, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_22 = tmp_21.view(64, -1, 2, 8)
        tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = torch.nn.functional.normalize(in_3, dim=-1)
        tmp_25 = torch.nn.functional.normalize(in_2, dim=-1)
        tmp_26 = tmp_25.transpose(-2, -1)
        tmp_25 = None
        tmp_27 = tmp_24 @ tmp_26
        tmp_24 = tmp_26 = None
        tmp_28 = torch.clamp(tmp_9, max=4.605170185988092)
        tmp_9 = None
        tmp_29 = tmp_28.exp()
        tmp_28 = None
        tmp_30 = tmp_27 * tmp_29
        tmp_27 = tmp_29 = None
        tmp_31 = torch.nn.functional.linear(tmp_2, tmp_5, tmp_4)
        tmp_2 = tmp_5 = tmp_4 = None
        tmp_32 = torch.nn.functional.relu(tmp_31, inplace=True)
        tmp_31 = None
        tmp_33 = torch.nn.functional.linear(tmp_32, tmp_6, None)
        tmp_32 = tmp_6 = None
        tmp_34 = tmp_33.view(-1, 2)
        tmp_33 = None
        tmp_35 = tmp_3.view(-1)
        tmp_3 = None
        tmp_36 = tmp_34[tmp_35]
        tmp_34 = tmp_35 = None
        tmp_37 = tmp_36.view(4, 4, -1)
        tmp_36 = None
        tmp_38 = tmp_37.permute(2, 0, 1)
        tmp_37 = None
        tmp_39 = tmp_38.contiguous()
        tmp_38 = None
        tmp_40 = torch.sigmoid(tmp_39)
        tmp_39 = None
        tmp_41 = 16 * tmp_40
        tmp_40 = None
        tmp_42 = tmp_41.unsqueeze(0)
        tmp_41 = None
        tmp_43 = tmp_30 + tmp_42
        tmp_30 = tmp_42 = None
        tmp_44 = torch.nn.functional.softmax(tmp_43, dim=-1)
        tmp_43 = None
        tmp_45 = torch.nn.functional.dropout(tmp_44, 0.0, False, False)
        tmp_44 = None
        tmp_46 = torch.matmul(tmp_45, tmp_23)
        tmp_45 = tmp_23 = None
        tmp_47 = tmp_46.permute(0, 2, 1, 3)
        tmp_46 = None
        tmp_48 = tmp_47.contiguous()
        tmp_47 = None
        tmp_49 = tmp_48.view((64, 4, 16))
        tmp_48 = None
        tmp_50 = torch.nn.functional.linear(tmp_49, tmp_1, tmp_0)
        tmp_49 = tmp_1 = tmp_0 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.0, False, False)
        tmp_50 = None
        tmp_52 = tmp_51.view(-1, 2, 2, 16)
        tmp_51 = None
        tmp_53 = tmp_52.view(-1, 8, 8, 2, 2, 16)
        tmp_52 = None
        tmp_54 = tmp_53.permute(0, 1, 3, 2, 4, 5)
        tmp_53 = None
        tmp_55 = tmp_54.contiguous()
        tmp_54 = None
        tmp_56 = tmp_55.view(-1, 16, 16, 16)
        tmp_55 = None
        tmp_57 = tmp_56.view(1, 256, 16)
        tmp_56 = None
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (16,), tmp_15, tmp_14, 1e-05)
        tmp_57 = tmp_15 = tmp_14 = None
        tmp_59 = in_0 + tmp_58
        tmp_58 = None
        tmp_60 = torch.nn.functional.linear(tmp_59, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_61 = torch.nn.functional.gelu(tmp_60)
        tmp_60 = None
        tmp_62 = torch.nn.functional.linear(tmp_61, tmp_17, tmp_16)
        tmp_61 = tmp_17 = tmp_16 = None
        tmp_63 = torch.nn.functional.dropout(tmp_62, 0.0, False, False)
        tmp_62 = None
        tmp_64 = torch.nn.functional.layer_norm(tmp_63, (16,), tmp_13, tmp_12, 1e-05)
        tmp_63 = tmp_13 = tmp_12 = None
        tmp_65 = tmp_59 + tmp_64
        tmp_59 = tmp_64 = None
        tmp_66 = tmp_65.view(1, 16, 16, 16)
        tmp_65 = None
        tmp_67 = tmp_66[slice(None, None, None), slice(0, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_68 = tmp_66[slice(None, None, None), slice(1, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_69 = tmp_66[slice(None, None, None), slice(0, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_70 = tmp_66[slice(None, None, None), slice(1, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_66 = None
        tmp_71 = torch.cat([tmp_67, tmp_68, tmp_69, tmp_70], -1)
        tmp_67 = tmp_68 = tmp_69 = tmp_70 = None
        tmp_72 = tmp_71.view(1, -1, 64)
        tmp_71 = None
        tmp_73 = torch.nn.functional.linear(tmp_72, tmp_20, None)
        tmp_72 = tmp_20 = None
        tmp_74 = torch.nn.functional.layer_norm(tmp_73, (32,), tmp_19, tmp_18, 1e-05)
        tmp_73 = tmp_19 = tmp_18 = None
        tmp_75 = tmp_74.view(1, 8, 8, 32)
        tmp_76 = torch.nn.functional.pad(tmp_75, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_75 = None
        tmp_77 = tmp_76.view(1, 4, 2, 4, 2, 32)
        tmp_76 = None
        tmp_78 = tmp_77.permute(0, 1, 3, 2, 4, 5)
        tmp_77 = None
        tmp_79 = tmp_78.contiguous()
        tmp_78 = None
        tmp_80 = tmp_79.view(-1, 2, 2, 32)
        tmp_79 = None
        tmp_81 = tmp_80.view(-1, 4, 32)
        tmp_80 = None
        return (tmp_81, tmp_74)