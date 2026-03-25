import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
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
        tmp_15 = torch.nn.functional.linear(in_16, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_16 = tmp_15.view(1, -1, 16, 64)
        tmp_15 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = tmp_0[slice(None, 729, None)]
        tmp_19 = tmp_18.reshape(1, 27, 27, -1)
        tmp_18 = None
        tmp_20 = tmp_19.permute(0, 3, 1, 2)
        tmp_19 = None
        tmp_21 = torch.nn.functional.interpolate(tmp_20, size=(27, 27), mode='bilinear')
        tmp_20 = None
        tmp_22 = tmp_21.permute(0, 2, 3, 1)
        tmp_21 = None
        tmp_23 = tmp_22.reshape(729, -1)
        tmp_22 = None
        tmp_24 = tmp_0[slice(729, None, None)]
        tmp_0 = None
        tmp_25 = torch.cat([tmp_23, tmp_24])
        tmp_23 = tmp_24 = None
        tmp_26 = torch.arange(14)
        tmp_27 = torch.arange(14)
        tmp_28 = torch.functional.meshgrid(tmp_26, tmp_27, indexing='ij')
        tmp_26 = tmp_27 = None
        tmp_29 = tmp_28[0]
        tmp_30 = tmp_28[1]
        tmp_28 = None
        tmp_31 = torch.stack((tmp_29, tmp_30))
        tmp_29 = tmp_30 = None
        tmp_32 = torch.flatten(tmp_31, 1)
        tmp_31 = None
        tmp_33 = tmp_32[slice(None, None, None), slice(None, None, None), None]
        tmp_34 = tmp_32[slice(None, None, None), None, slice(None, None, None)]
        tmp_32 = None
        tmp_35 = tmp_33 - tmp_34
        tmp_33 = tmp_34 = None
        tmp_36 = tmp_35.permute(1, 2, 0)
        tmp_35 = None
        tmp_37 = tmp_36.contiguous()
        tmp_36 = None
        tmp_38 = tmp_37[slice(None, None, None), slice(None, None, None), 0]
        tmp_38 += 13
        tmp_39 = tmp_38
        tmp_38 = None
        tmp_37[slice(None, None, None), slice(None, None, None), 0] = tmp_39
        tmp_40 = tmp_37
        tmp_39 = tmp_40 = None
        tmp_41 = tmp_37[slice(None, None, None), slice(None, None, None), 1]
        tmp_41 += 13
        tmp_42 = tmp_41
        tmp_41 = None
        tmp_37[slice(None, None, None), slice(None, None, None), 1] = tmp_42
        tmp_43 = tmp_37
        tmp_42 = tmp_43 = None
        tmp_44 = tmp_37[slice(None, None, None), slice(None, None, None), 0]
        tmp_44 *= 27
        tmp_45 = tmp_44
        tmp_44 = None
        tmp_37[slice(None, None, None), slice(None, None, None), 0] = tmp_45
        tmp_46 = tmp_37
        tmp_45 = tmp_46 = None
        tmp_47 = torch.zeros(size=(197, 197), dtype=torch.int64)
        tmp_48 = tmp_37.sum(-1)
        tmp_37 = None
        tmp_47[slice(1, None, None), slice(1, None, None)] = tmp_48
        tmp_49 = tmp_47
        tmp_48 = tmp_49 = None
        tmp_47[0, slice(0, None, None)] = 729
        tmp_50 = tmp_47
        tmp_50 = None
        tmp_47[slice(0, None, None), 0] = 730
        tmp_51 = tmp_47
        tmp_51 = None
        tmp_47[0, 0] = 731
        tmp_52 = tmp_47
        tmp_52 = None
        tmp_53 = tmp_47.view(-1)
        tmp_47 = None
        tmp_54 = tmp_25[tmp_53]
        tmp_25 = tmp_53 = None
        tmp_55 = tmp_54.view(197, 197, -1)
        tmp_54 = None
        tmp_56 = tmp_55.permute(2, 0, 1)
        tmp_55 = None
        tmp_57 = tmp_56.contiguous()
        tmp_56 = None
        tmp_58 = tmp_57.unsqueeze(0)
        tmp_57 = None
        tmp_59 = torch.nn.functional.scaled_dot_product_attention(in_18, in_15, tmp_17, attn_mask=tmp_58, dropout_p=0.0, is_causal=False, scale=0.125)
        tmp_17 = tmp_58 = None
        tmp_60 = tmp_59.permute(0, 2, 1, 3)
        tmp_59 = None
        tmp_61 = tmp_60.contiguous()
        tmp_60 = None
        tmp_62 = tmp_61.view(1, 197, 1024)
        tmp_61 = None
        tmp_63 = torch.nn.functional.linear(tmp_62, tmp_4, tmp_3)
        tmp_62 = tmp_4 = tmp_3 = None
        tmp_64 = torch.nn.functional.dropout(tmp_63, 0.0, False, False)
        tmp_63 = None
        tmp_65 = tmp_11 * tmp_64
        tmp_11 = tmp_64 = None
        tmp_66 = tmp_65 + in_17
        tmp_65 = None
        tmp_67 = torch.nn.functional.layer_norm(tmp_66, (1024,), tmp_8, tmp_7, 1e-12)
        tmp_8 = tmp_7 = None
        tmp_68 = torch.nn.functional.linear(tmp_67, tmp_6, tmp_5)
        tmp_67 = tmp_6 = tmp_5 = None
        tmp_69 = torch.nn.functional.gelu(tmp_68)
        tmp_68 = None
        tmp_70 = torch.nn.functional.linear(tmp_69, tmp_10, tmp_9)
        tmp_69 = tmp_10 = tmp_9 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.0, False, False)
        tmp_70 = None
        tmp_72 = tmp_12 * tmp_71
        tmp_12 = tmp_71 = None
        tmp_73 = tmp_72 + tmp_66
        tmp_72 = tmp_66 = None
        tmp_74 = tmp_73[slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_75 = tmp_74.mean(1)
        tmp_74 = None
        tmp_76 = torch.nn.functional.layer_norm(tmp_75, (1024,), tmp_14, tmp_13, 1e-12)
        tmp_75 = tmp_14 = tmp_13 = None
        return (tmp_73, tmp_76)