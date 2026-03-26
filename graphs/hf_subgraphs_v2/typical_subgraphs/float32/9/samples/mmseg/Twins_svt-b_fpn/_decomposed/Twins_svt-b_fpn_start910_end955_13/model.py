import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, in_0, in_1, in_2, in_3, in_4):
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
        tmp_21 = w_21
        tmp_22 = w_22
        tmp_23 = w_23
        tmp_24 = w_24
        tmp_25 = w_25
        tmp_26 = w_26
        tmp_27 = w_27
        tmp_28 = w_28
        tmp_29 = w_29
        tmp_30 = w_30
        tmp_31 = w_31
        tmp_32 = w_32
        tmp_33 = w_33
        tmp_34 = torch.nn.functional.gelu(in_0, approximate='none')
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.0, False, False)
        tmp_34 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_1, tmp_0)
        tmp_35 = tmp_1 = tmp_0 = None
        tmp_37 = torch.nn.functional.dropout(tmp_36, 0.0, False, False)
        tmp_36 = None
        tmp_38 = in_2 + tmp_37
        tmp_37 = None
        tmp_39 = tmp_38.transpose(1, 2)
        tmp_38 = None
        tmp_40 = tmp_39.view(1, 768, 16, 16)
        tmp_39 = None
        tmp_41 = torch.conv2d(tmp_40, tmp_17, tmp_16, (1, 1), (1, 1), (1, 1), 768)
        tmp_17 = tmp_16 = None
        tmp_42 = tmp_41 + tmp_40
        tmp_41 = tmp_40 = None
        tmp_43 = tmp_42.flatten(2)
        tmp_42 = None
        tmp_44 = tmp_43.transpose(1, 2)
        tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (768,), tmp_11, tmp_10, 1e-05)
        tmp_11 = tmp_10 = None
        tmp_46 = tmp_45.transpose(0, 1)
        tmp_47 = tmp_45.transpose(0, 1)
        tmp_45 = None
        tmp_48 = torch.nn.functional.multi_head_attention_forward(tmp_46, tmp_47, tmp_47, 768, 24, tmp_5, tmp_4, None, None, False, 0.0, tmp_3, tmp_2, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_46 = tmp_47 = tmp_5 = tmp_4 = tmp_3 = tmp_2 = None
        tmp_49 = tmp_48[0]
        tmp_48 = None
        tmp_50 = tmp_49.transpose(0, 1)
        tmp_49 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.0, False, False)
        tmp_50 = None
        tmp_52 = 0.0 + tmp_51
        tmp_51 = None
        tmp_53 = tmp_44 + tmp_52
        tmp_44 = tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (768,), tmp_13, tmp_12, 1e-05)
        tmp_13 = tmp_12 = None
        tmp_55 = torch.nn.functional.linear(tmp_54, tmp_7, tmp_6)
        tmp_54 = tmp_7 = tmp_6 = None
        tmp_56 = torch.nn.functional.gelu(tmp_55, approximate='none')
        tmp_55 = None
        tmp_57 = torch.nn.functional.dropout(tmp_56, 0.0, False, False)
        tmp_56 = None
        tmp_58 = torch.nn.functional.linear(tmp_57, tmp_9, tmp_8)
        tmp_57 = tmp_9 = tmp_8 = None
        tmp_59 = torch.nn.functional.dropout(tmp_58, 0.0, False, False)
        tmp_58 = None
        tmp_60 = tmp_53 + tmp_59
        tmp_53 = tmp_59 = None
        tmp_61 = torch.nn.functional.layer_norm(tmp_60, (768,), tmp_15, tmp_14, 1e-05)
        tmp_60 = tmp_15 = tmp_14 = None
        tmp_62 = tmp_61.reshape(1, 16, 16, -1)
        tmp_61 = None
        tmp_63 = tmp_62.permute(0, 3, 1, 2)
        tmp_62 = None
        tmp_64 = tmp_63.contiguous()
        tmp_63 = None
        tmp_65 = torch.conv2d(in_3, tmp_27, tmp_26, (1, 1), (0, 0), (1, 1), 1)
        tmp_27 = tmp_26 = None
        tmp_66 = torch.conv2d(in_4, tmp_29, tmp_28, (1, 1), (0, 0), (1, 1), 1)
        tmp_29 = tmp_28 = None
        tmp_67 = torch.conv2d(in_1, tmp_31, tmp_30, (1, 1), (0, 0), (1, 1), 1)
        tmp_31 = tmp_30 = None
        tmp_68 = torch.conv2d(tmp_64, tmp_33, tmp_32, (1, 1), (0, 0), (1, 1), 1)
        tmp_64 = tmp_33 = tmp_32 = None
        tmp_69 = torch.nn.functional.interpolate(tmp_68, (32, 32), None, 'nearest', None)
        tmp_70 = tmp_67 + tmp_69
        tmp_67 = tmp_69 = None
        tmp_71 = torch.nn.functional.interpolate(tmp_70, (64, 64), None, 'nearest', None)
        tmp_72 = tmp_66 + tmp_71
        tmp_66 = tmp_71 = None
        tmp_73 = torch.nn.functional.interpolate(tmp_72, (128, 128), None, 'nearest', None)
        tmp_74 = tmp_65 + tmp_73
        tmp_65 = tmp_73 = None
        tmp_75 = torch.conv2d(tmp_74, tmp_19, tmp_18, (1, 1), (1, 1), (1, 1), 1)
        tmp_74 = tmp_19 = tmp_18 = None
        tmp_76 = torch.conv2d(tmp_72, tmp_21, tmp_20, (1, 1), (1, 1), (1, 1), 1)
        tmp_72 = tmp_21 = tmp_20 = None
        tmp_77 = torch.conv2d(tmp_70, tmp_23, tmp_22, (1, 1), (1, 1), (1, 1), 1)
        tmp_70 = tmp_23 = tmp_22 = None
        tmp_78 = torch.conv2d(tmp_68, tmp_25, tmp_24, (1, 1), (1, 1), (1, 1), 1)
        tmp_68 = tmp_25 = tmp_24 = None
        return (tmp_75, tmp_76, tmp_77, tmp_78)