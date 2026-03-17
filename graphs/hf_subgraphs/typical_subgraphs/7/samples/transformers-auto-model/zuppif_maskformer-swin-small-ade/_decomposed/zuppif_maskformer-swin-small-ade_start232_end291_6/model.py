import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21):
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
        tmp_17 = torch.nn.functional.linear(in_18, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_18 = tmp_17.view((100, 49, -1, 32))
        tmp_17 = None
        tmp_19 = tmp_18.transpose(1, 2)
        tmp_18 = None
        tmp_20 = in_19.transpose(-1, -2)
        tmp_21 = torch.matmul(in_21, tmp_20)
        tmp_20 = None
        tmp_22 = tmp_21 / 5.656854249492381
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
        tmp_30 = tmp_29.view(1, 100, 6, 49, 49)
        tmp_29 = None
        tmp_31 = in_17.unsqueeze(1)
        tmp_32 = tmp_31.unsqueeze(0)
        tmp_31 = None
        tmp_33 = tmp_30 + tmp_32
        tmp_30 = tmp_32 = None
        tmp_34 = tmp_33.view(-1, 6, 49, 49)
        tmp_33 = None
        tmp_35 = torch.nn.functional.softmax(tmp_34, dim=-1)
        tmp_34 = None
        tmp_36 = torch.nn.functional.dropout(tmp_35, 0.0, False, False)
        tmp_35 = None
        tmp_37 = torch.matmul(tmp_36, tmp_19)
        tmp_36 = tmp_19 = None
        tmp_38 = tmp_37.permute(0, 2, 1, 3)
        tmp_37 = None
        tmp_39 = tmp_38.contiguous()
        tmp_38 = None
        tmp_40 = tmp_39.view((100, 49, 192))
        tmp_39 = None
        tmp_41 = torch.nn.functional.linear(tmp_40, tmp_1, tmp_0)
        tmp_40 = tmp_1 = tmp_0 = None
        tmp_42 = torch.nn.functional.dropout(tmp_41, 0.0, False, False)
        tmp_41 = None
        tmp_43 = tmp_42.view(-1, 7, 7, 192)
        tmp_42 = None
        tmp_44 = tmp_43.view(-1, 10, 10, 7, 7, 192)
        tmp_43 = None
        tmp_45 = tmp_44.permute(0, 1, 3, 2, 4, 5)
        tmp_44 = None
        tmp_46 = tmp_45.contiguous()
        tmp_45 = None
        tmp_47 = tmp_46.view(-1, 70, 70, 192)
        tmp_46 = None
        tmp_48 = torch.roll(tmp_47, shifts=(3, 3), dims=(1, 2))
        tmp_47 = None
        tmp_49 = tmp_48[slice(None, None, None), slice(None, 64, None), slice(None, 64, None), slice(None, None, None)]
        tmp_48 = None
        tmp_50 = tmp_49.contiguous()
        tmp_49 = None
        tmp_51 = tmp_50.view(1, 4096, 192)
        tmp_50 = None
        tmp_52 = in_20 + tmp_51
        tmp_51 = None
        tmp_53 = torch.nn.functional.layer_norm(tmp_52, (192,), tmp_9, tmp_8, 1e-05)
        tmp_9 = tmp_8 = None
        tmp_54 = torch.nn.functional.linear(tmp_53, tmp_7, tmp_6)
        tmp_53 = tmp_7 = tmp_6 = None
        tmp_55 = torch.nn.functional.gelu(tmp_54)
        tmp_54 = None
        tmp_56 = torch.nn.functional.linear(tmp_55, tmp_11, tmp_10)
        tmp_55 = tmp_11 = tmp_10 = None
        tmp_57 = torch.nn.functional.dropout(tmp_56, 0.0, False, False)
        tmp_56 = None
        tmp_58 = tmp_52 + tmp_57
        tmp_52 = tmp_57 = None
        tmp_59 = tmp_58.view(1, 64, 64, 192)
        tmp_60 = tmp_59[slice(None, None, None), slice(0, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_61 = tmp_59[slice(None, None, None), slice(1, None, 2), slice(0, None, 2), slice(None, None, None)]
        tmp_62 = tmp_59[slice(None, None, None), slice(0, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_63 = tmp_59[slice(None, None, None), slice(1, None, 2), slice(1, None, 2), slice(None, None, None)]
        tmp_59 = None
        tmp_64 = torch.cat([tmp_60, tmp_61, tmp_62, tmp_63], -1)
        tmp_60 = tmp_61 = tmp_62 = tmp_63 = None
        tmp_65 = tmp_64.view(1, -1, 768)
        tmp_64 = None
        tmp_66 = torch.nn.functional.layer_norm(tmp_65, (768,), tmp_13, tmp_12, 1e-05)
        tmp_65 = tmp_13 = tmp_12 = None
        tmp_67 = torch.nn.functional.linear(tmp_66, tmp_14, None)
        tmp_66 = tmp_14 = None
        tmp_68 = torch.nn.functional.layer_norm(tmp_67, (384,), tmp_16, tmp_15, 1e-05)
        tmp_16 = tmp_15 = None
        tmp_69 = tmp_68.view(1, 32, 32, 384)
        tmp_68 = None
        tmp_70 = torch.nn.functional.pad(tmp_69, (0, 0, 0, 3, 0, 3), 'constant', None)
        tmp_69 = None
        tmp_71 = tmp_70.view(1, 5, 7, 5, 7, 384)
        tmp_70 = None
        tmp_72 = tmp_71.permute(0, 1, 3, 2, 4, 5)
        tmp_71 = None
        tmp_73 = tmp_72.contiguous()
        tmp_72 = None
        tmp_74 = tmp_73.view(-1, 7, 7, 384)
        tmp_73 = None
        tmp_75 = tmp_74.view(-1, 49, 384)
        tmp_74 = None
        return (tmp_75, tmp_67, tmp_58)