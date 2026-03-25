import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, in_0, in_1, in_2, in_3, in_4):
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
        tmp_15 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_16 = tmp_15.view(1, -1, 16, 64)
        tmp_15 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = torch.nn.functional.scaled_dot_product_attention(in_3, in_0, tmp_17, attn_mask=in_4, dropout_p=0.0, is_causal=False, scale=0.125)
        tmp_17 = None
        tmp_19 = tmp_18.permute(0, 2, 1, 3)
        tmp_18 = None
        tmp_20 = tmp_19.contiguous()
        tmp_19 = None
        tmp_21 = tmp_20.view(1, 197, 1024)
        tmp_20 = None
        tmp_22 = torch.nn.functional.linear(tmp_21, tmp_5, tmp_4)
        tmp_21 = tmp_5 = tmp_4 = None
        tmp_23 = torch.nn.functional.dropout(tmp_22, 0.0, False, False)
        tmp_22 = None
        tmp_24 = tmp_12 * tmp_23
        tmp_12 = tmp_23 = None
        tmp_25 = tmp_24 + in_2
        tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (1024,), tmp_9, tmp_8, 1e-12)
        tmp_9 = tmp_8 = None
        tmp_27 = torch.nn.functional.linear(tmp_26, tmp_7, tmp_6)
        tmp_26 = tmp_7 = tmp_6 = None
        tmp_28 = torch.nn.functional.gelu(tmp_27)
        tmp_27 = None
        tmp_29 = torch.nn.functional.linear(tmp_28, tmp_11, tmp_10)
        tmp_28 = tmp_11 = tmp_10 = None
        tmp_30 = torch.nn.functional.dropout(tmp_29, 0.0, False, False)
        tmp_29 = None
        tmp_31 = tmp_13 * tmp_30
        tmp_13 = tmp_30 = None
        tmp_32 = tmp_31 + tmp_25
        tmp_31 = tmp_25 = None
        tmp_33 = tmp_14[slice(None, 729, None)]
        tmp_34 = tmp_33.reshape(1, 27, 27, -1)
        tmp_33 = None
        tmp_35 = tmp_34.permute(0, 3, 1, 2)
        tmp_34 = None
        tmp_36 = torch.nn.functional.interpolate(tmp_35, size=(27, 27), mode='bilinear')
        tmp_35 = None
        tmp_37 = tmp_36.permute(0, 2, 3, 1)
        tmp_36 = None
        tmp_38 = tmp_37.reshape(729, -1)
        tmp_37 = None
        tmp_39 = tmp_14[slice(729, None, None)]
        tmp_14 = None
        tmp_40 = torch.cat([tmp_38, tmp_39])
        tmp_38 = tmp_39 = None
        tmp_41 = torch.arange(14)
        tmp_42 = torch.arange(14)
        tmp_43 = torch.functional.meshgrid(tmp_41, tmp_42, indexing='ij')
        tmp_41 = tmp_42 = None
        tmp_44 = tmp_43[0]
        tmp_45 = tmp_43[1]
        tmp_43 = None
        tmp_46 = torch.stack((tmp_44, tmp_45))
        tmp_44 = tmp_45 = None
        tmp_47 = torch.flatten(tmp_46, 1)
        tmp_46 = None
        tmp_48 = tmp_47[slice(None, None, None), slice(None, None, None), None]
        tmp_49 = tmp_47[slice(None, None, None), None, slice(None, None, None)]
        tmp_47 = None
        tmp_50 = tmp_48 - tmp_49
        tmp_48 = tmp_49 = None
        tmp_51 = tmp_50.permute(1, 2, 0)
        tmp_50 = None
        tmp_52 = tmp_51.contiguous()
        tmp_51 = None
        tmp_53 = tmp_52[slice(None, None, None), slice(None, None, None), 0]
        tmp_53 += 13
        tmp_54 = tmp_53
        tmp_53 = None
        tmp_52[slice(None, None, None), slice(None, None, None), 0] = tmp_54
        tmp_55 = tmp_52
        tmp_54 = tmp_55 = None
        tmp_56 = tmp_52[slice(None, None, None), slice(None, None, None), 1]
        tmp_56 += 13
        tmp_57 = tmp_56
        tmp_56 = None
        tmp_52[slice(None, None, None), slice(None, None, None), 1] = tmp_57
        tmp_58 = tmp_52
        tmp_57 = tmp_58 = None
        tmp_59 = tmp_52[slice(None, None, None), slice(None, None, None), 0]
        tmp_59 *= 27
        tmp_60 = tmp_59
        tmp_59 = None
        tmp_52[slice(None, None, None), slice(None, None, None), 0] = tmp_60
        tmp_61 = tmp_52
        tmp_60 = tmp_61 = None
        tmp_62 = torch.zeros(size=(197, 197), dtype=torch.int64)
        tmp_63 = tmp_52.sum(-1)
        tmp_52 = None
        tmp_62[slice(1, None, None), slice(1, None, None)] = tmp_63
        tmp_64 = tmp_62
        tmp_63 = tmp_64 = None
        tmp_62[0, slice(0, None, None)] = 729
        tmp_65 = tmp_62
        tmp_65 = None
        tmp_62[slice(0, None, None), 0] = 730
        tmp_66 = tmp_62
        tmp_66 = None
        tmp_62[0, 0] = 731
        tmp_67 = tmp_62
        tmp_67 = None
        tmp_68 = tmp_62.view(-1)
        tmp_62 = None
        tmp_69 = tmp_40[tmp_68]
        tmp_40 = tmp_68 = None
        tmp_70 = tmp_69.view(197, 197, -1)
        tmp_69 = None
        tmp_71 = tmp_70.permute(2, 0, 1)
        tmp_70 = None
        tmp_72 = tmp_71.contiguous()
        tmp_71 = None
        tmp_73 = tmp_72.unsqueeze(0)
        tmp_72 = None
        tmp_74 = torch.nn.functional.layer_norm(tmp_32, (1024,), tmp_1, tmp_0, 1e-12)
        tmp_1 = tmp_0 = None
        return (tmp_74, tmp_32, tmp_73)