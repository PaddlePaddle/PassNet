import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50):
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
        tmp_17 = in_17
        tmp_18 = in_18
        tmp_19 = in_19
        tmp_20 = in_20
        tmp_21 = in_21
        tmp_22 = in_22
        tmp_23 = in_23
        tmp_24 = in_24
        tmp_25 = in_25
        tmp_26 = in_26
        tmp_27 = in_27
        tmp_28 = in_28
        tmp_29 = in_29
        tmp_30 = in_30
        tmp_31 = in_31
        tmp_32 = in_32
        tmp_33 = in_33
        tmp_34 = in_34
        tmp_35 = in_35
        tmp_36 = in_36
        tmp_37 = in_37
        tmp_38 = in_38
        tmp_39 = in_39
        tmp_40 = in_40
        tmp_41 = in_41
        tmp_42 = in_42
        tmp_43 = in_43
        tmp_44 = in_44
        tmp_45 = in_45
        tmp_46 = in_46
        tmp_47 = in_47
        tmp_48 = in_48
        tmp_49 = torch.nn.functional.linear(in_49, tmp_19, tmp_18)
        tmp_19 = tmp_18 = None
        tmp_50 = in_50 + tmp_49
        tmp_49 = None
        tmp_51 = torch.nn.functional.layer_norm(tmp_50, (1536,), tmp_13, tmp_12, 1e-06)
        tmp_13 = tmp_12 = None
        tmp_52 = torch.nn.functional.linear(tmp_51, tmp_15, tmp_14)
        tmp_51 = tmp_15 = tmp_14 = None
        tmp_53 = torch.nn.functional.gelu(tmp_52, approximate='tanh')
        tmp_52 = None
        tmp_54 = torch.nn.functional.linear(tmp_53, tmp_17, tmp_16)
        tmp_53 = tmp_17 = tmp_16 = None
        tmp_55 = tmp_50 + tmp_54
        tmp_50 = tmp_54 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (1536,), tmp_21, tmp_20, 1e-06)
        tmp_21 = tmp_20 = None
        tmp_57 = torch.nn.functional.linear(tmp_56, tmp_33, tmp_32)
        tmp_33 = tmp_32 = None
        tmp_58 = torch.nn.functional.linear(tmp_56, tmp_29, tmp_28)
        tmp_29 = tmp_28 = None
        tmp_59 = torch.nn.functional.linear(tmp_56, tmp_35, tmp_34)
        tmp_56 = tmp_35 = tmp_34 = None
        tmp_60 = tmp_57.view(1, 576, 16, 96)
        tmp_57 = None
        tmp_61 = tmp_60.transpose(1, 2)
        tmp_60 = None
        tmp_62 = tmp_58.view(1, 576, 16, 96)
        tmp_58 = None
        tmp_63 = tmp_62.transpose(1, 2)
        tmp_62 = None
        tmp_64 = tmp_59.view(1, 576, 16, 96)
        tmp_59 = None
        tmp_65 = tmp_64.transpose(1, 2)
        tmp_64 = None
        tmp_66 = tmp_61.contiguous()
        tmp_61 = None
        tmp_67 = tmp_63.contiguous()
        tmp_63 = None
        tmp_68 = tmp_65.contiguous()
        tmp_65 = None
        tmp_69 = torch.nn.functional.scaled_dot_product_attention(tmp_66, tmp_67, tmp_68, attn_mask=None, dropout_p=0.0, scale=0.10206207261596575, is_causal=False)
        tmp_66 = tmp_67 = tmp_68 = None
        tmp_70 = tmp_69.transpose(1, 2)
        tmp_69 = None
        tmp_71 = tmp_70.contiguous()
        tmp_70 = None
        tmp_72 = tmp_71.reshape(1, 576, 1536)
        tmp_71 = None
        tmp_73 = tmp_72.contiguous()
        tmp_72 = None
        tmp_74 = torch.nn.functional.linear(tmp_73, tmp_31, tmp_30)
        tmp_73 = tmp_31 = tmp_30 = None
        tmp_75 = tmp_55 + tmp_74
        tmp_55 = tmp_74 = None
        tmp_76 = torch.nn.functional.layer_norm(tmp_75, (1536,), tmp_23, tmp_22, 1e-06)
        tmp_23 = tmp_22 = None
        tmp_77 = torch.nn.functional.linear(tmp_76, tmp_25, tmp_24)
        tmp_76 = tmp_25 = tmp_24 = None
        tmp_78 = torch.nn.functional.gelu(tmp_77, approximate='tanh')
        tmp_77 = None
        tmp_79 = torch.nn.functional.linear(tmp_78, tmp_27, tmp_26)
        tmp_78 = tmp_27 = tmp_26 = None
        tmp_80 = tmp_75 + tmp_79
        tmp_75 = tmp_79 = None
        tmp_81 = torch.nn.functional.layer_norm(tmp_80, (1536,), tmp_48, tmp_47, 1e-06)
        tmp_80 = tmp_48 = tmp_47 = None
        tmp_82 = tmp_46.repeat(1, 1, 1)
        tmp_46 = None
        tmp_83 = tmp_82.transpose(1, 0)
        tmp_82 = None
        tmp_84 = tmp_81.transpose(1, 0)
        tmp_85 = torch.nn.functional.multi_head_attention_forward(tmp_83, tmp_84, tmp_84, 1536, 16, tmp_39, tmp_38, None, None, False, 0.0, tmp_37, tmp_36, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_83 = tmp_84 = tmp_39 = tmp_38 = tmp_37 = tmp_36 = None
        tmp_86 = tmp_85[0]
        tmp_85 = None
        tmp_87 = tmp_86.transpose(1, 0)
        tmp_86 = None
        tmp_88 = torch.nn.functional.layer_norm(tmp_87, (1536,), tmp_41, tmp_40, 1e-06)
        tmp_41 = tmp_40 = None
        tmp_89 = torch.nn.functional.linear(tmp_88, tmp_43, tmp_42)
        tmp_88 = tmp_43 = tmp_42 = None
        tmp_90 = torch.nn.functional.gelu(tmp_89, approximate='tanh')
        tmp_89 = None
        tmp_91 = torch.nn.functional.linear(tmp_90, tmp_45, tmp_44)
        tmp_90 = tmp_45 = tmp_44 = None
        tmp_92 = tmp_87 + tmp_91
        tmp_87 = tmp_91 = None
        tmp_93 = tmp_92[slice(None, None, None), 0]
        tmp_92 = None
        tmp_94 = tmp_0.view(-1, 7)
        tmp_0 = None
        tmp_95 = tmp_1[slice(None, None, None), slice(None, 7, None)]
        tmp_1 = None
        tmp_96 = torch.nn.functional.embedding(tmp_94, tmp_3, None, None, 2.0, False, False)
        tmp_94 = tmp_3 = None
        tmp_97 = torch.nn.functional.embedding(tmp_95, tmp_2, None, None, 2.0, False, False)
        tmp_95 = tmp_2 = None
        tmp_98 = tmp_96 + tmp_97
        tmp_96 = tmp_97 = None
        tmp_99 = torch.nn.functional.layer_norm(tmp_98, (1152,), tmp_5, tmp_4, 1e-06)
        tmp_5 = tmp_4 = None
        tmp_100 = torch.nn.functional.linear(tmp_99, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_101 = torch.nn.functional.linear(tmp_99, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_102 = torch.nn.functional.linear(tmp_99, tmp_11, tmp_10)
        tmp_99 = tmp_11 = tmp_10 = None
        return (tmp_98, tmp_101, tmp_81, tmp_93, tmp_100, tmp_102)