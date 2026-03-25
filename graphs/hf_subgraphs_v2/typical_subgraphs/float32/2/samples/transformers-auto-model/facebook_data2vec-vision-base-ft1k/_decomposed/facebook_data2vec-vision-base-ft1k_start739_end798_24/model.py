import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16):
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
        tmp_13 = torch.nn.functional.linear(in_14, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_14 = tmp_13.view(1, -1, 12, 64)
        tmp_13 = None
        tmp_15 = tmp_14.transpose(1, 2)
        tmp_14 = None
        tmp_16 = tmp_0[slice(None, 729, None)]
        tmp_17 = tmp_16.reshape(1, 27, 27, -1)
        tmp_16 = None
        tmp_18 = tmp_17.permute(0, 3, 1, 2)
        tmp_17 = None
        tmp_19 = torch.nn.functional.interpolate(tmp_18, size=(27, 27), mode='bilinear')
        tmp_18 = None
        tmp_20 = tmp_19.permute(0, 2, 3, 1)
        tmp_19 = None
        tmp_21 = tmp_20.reshape(729, -1)
        tmp_20 = None
        tmp_22 = tmp_0[slice(729, None, None)]
        tmp_0 = None
        tmp_23 = torch.cat([tmp_21, tmp_22])
        tmp_21 = tmp_22 = None
        tmp_24 = torch.arange(14)
        tmp_25 = torch.arange(14)
        tmp_26 = torch.functional.meshgrid(tmp_24, tmp_25, indexing='ij')
        tmp_24 = tmp_25 = None
        tmp_27 = tmp_26[0]
        tmp_28 = tmp_26[1]
        tmp_26 = None
        tmp_29 = torch.stack((tmp_27, tmp_28))
        tmp_27 = tmp_28 = None
        tmp_30 = torch.flatten(tmp_29, 1)
        tmp_29 = None
        tmp_31 = tmp_30[slice(None, None, None), slice(None, None, None), None]
        tmp_32 = tmp_30[slice(None, None, None), None, slice(None, None, None)]
        tmp_30 = None
        tmp_33 = tmp_31 - tmp_32
        tmp_31 = tmp_32 = None
        tmp_34 = tmp_33.permute(1, 2, 0)
        tmp_33 = None
        tmp_35 = tmp_34.contiguous()
        tmp_34 = None
        tmp_36 = tmp_35[slice(None, None, None), slice(None, None, None), 0]
        tmp_36 += 13
        tmp_37 = tmp_36
        tmp_36 = None
        tmp_35[slice(None, None, None), slice(None, None, None), 0] = tmp_37
        tmp_38 = tmp_35
        tmp_37 = tmp_38 = None
        tmp_39 = tmp_35[slice(None, None, None), slice(None, None, None), 1]
        tmp_39 += 13
        tmp_40 = tmp_39
        tmp_39 = None
        tmp_35[slice(None, None, None), slice(None, None, None), 1] = tmp_40
        tmp_41 = tmp_35
        tmp_40 = tmp_41 = None
        tmp_42 = tmp_35[slice(None, None, None), slice(None, None, None), 0]
        tmp_42 *= 27
        tmp_43 = tmp_42
        tmp_42 = None
        tmp_35[slice(None, None, None), slice(None, None, None), 0] = tmp_43
        tmp_44 = tmp_35
        tmp_43 = tmp_44 = None
        tmp_45 = torch.zeros(size=(197, 197), dtype=torch.int64)
        tmp_46 = tmp_35.sum(-1)
        tmp_35 = None
        tmp_45[slice(1, None, None), slice(1, None, None)] = tmp_46
        tmp_47 = tmp_45
        tmp_46 = tmp_47 = None
        tmp_45[0, slice(0, None, None)] = 729
        tmp_48 = tmp_45
        tmp_48 = None
        tmp_45[slice(0, None, None), 0] = 730
        tmp_49 = tmp_45
        tmp_49 = None
        tmp_45[0, 0] = 731
        tmp_50 = tmp_45
        tmp_50 = None
        tmp_51 = tmp_45.view(-1)
        tmp_45 = None
        tmp_52 = tmp_23[tmp_51]
        tmp_23 = tmp_51 = None
        tmp_53 = tmp_52.view(197, 197, -1)
        tmp_52 = None
        tmp_54 = tmp_53.permute(2, 0, 1)
        tmp_53 = None
        tmp_55 = tmp_54.contiguous()
        tmp_54 = None
        tmp_56 = tmp_55.unsqueeze(0)
        tmp_55 = None
        tmp_57 = torch.nn.functional.scaled_dot_product_attention(in_16, in_13, tmp_15, attn_mask=tmp_56, dropout_p=0.0, is_causal=False, scale=0.125)
        tmp_15 = tmp_56 = None
        tmp_58 = tmp_57.permute(0, 2, 1, 3)
        tmp_57 = None
        tmp_59 = tmp_58.contiguous()
        tmp_58 = None
        tmp_60 = tmp_59.view(1, 197, 768)
        tmp_59 = None
        tmp_61 = torch.nn.functional.linear(tmp_60, tmp_4, tmp_3)
        tmp_60 = tmp_4 = tmp_3 = None
        tmp_62 = torch.nn.functional.dropout(tmp_61, 0.0, False, False)
        tmp_61 = None
        tmp_63 = tmp_11 * tmp_62
        tmp_11 = tmp_62 = None
        tmp_64 = tmp_63 + in_15
        tmp_63 = None
        tmp_65 = torch.nn.functional.layer_norm(tmp_64, (768,), tmp_8, tmp_7, 1e-12)
        tmp_8 = tmp_7 = None
        tmp_66 = torch.nn.functional.linear(tmp_65, tmp_6, tmp_5)
        tmp_65 = tmp_6 = tmp_5 = None
        tmp_67 = torch.nn.functional.gelu(tmp_66)
        tmp_66 = None
        tmp_68 = torch.nn.functional.linear(tmp_67, tmp_10, tmp_9)
        tmp_67 = tmp_10 = tmp_9 = None
        tmp_69 = torch.nn.functional.dropout(tmp_68, 0.0, False, False)
        tmp_68 = None
        tmp_70 = tmp_12 * tmp_69
        tmp_12 = tmp_69 = None
        tmp_71 = tmp_70 + tmp_64
        tmp_70 = tmp_64 = None
        return (tmp_71,)