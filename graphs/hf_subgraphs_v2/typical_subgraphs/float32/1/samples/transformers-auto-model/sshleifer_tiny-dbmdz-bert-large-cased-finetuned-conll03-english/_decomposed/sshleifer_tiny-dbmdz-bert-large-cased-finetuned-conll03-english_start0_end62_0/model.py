import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42):
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
        tmp_43 = tmp_2[slice(None, None, None), slice(0, 512, None)]
        tmp_2 = None
        tmp_44 = torch.nn.functional.embedding(tmp_1, tmp_7, 0, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_45 = torch.nn.functional.embedding(tmp_42, tmp_6, None, None, 2.0, False, False)
        tmp_42 = tmp_6 = None
        tmp_46 = tmp_44 + tmp_45
        tmp_44 = tmp_45 = None
        tmp_47 = torch.nn.functional.embedding(tmp_43, tmp_5, None, None, 2.0, False, False)
        tmp_43 = tmp_5 = None
        tmp_46 += tmp_47
        tmp_48 = tmp_46
        tmp_46 = tmp_47 = None
        tmp_49 = torch.nn.functional.layer_norm(tmp_48, (2,), tmp_4, tmp_3, 1e-12)
        tmp_48 = tmp_4 = tmp_3 = None
        tmp_50 = torch.nn.functional.dropout(tmp_49, 0.1, False, False)
        tmp_49 = None
        tmp_51 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_52 = tmp_51.expand(1, 1, 512, 512)
        tmp_51 = None
        tmp_53 = tmp_52.to(torch.float32)
        tmp_52 = None
        tmp_54 = torch.tensor(1.0, dtype=torch.float32)
        tmp_55 = tmp_54 - tmp_53
        tmp_54 = tmp_53 = None
        tmp_56 = tmp_55.to(torch.bool)
        tmp_57 = tmp_55.masked_fill(tmp_56, -3.4028234663852886e+38)
        tmp_55 = tmp_56 = None
        tmp_58 = torch.nn.functional.linear(tmp_50, tmp_15, tmp_14)
        tmp_15 = tmp_14 = None
        tmp_59 = tmp_58.view(1, -1, 2, 1)
        tmp_58 = None
        tmp_60 = tmp_59.transpose(1, 2)
        tmp_59 = None
        tmp_61 = torch.nn.functional.linear(tmp_50, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_62 = tmp_61.view(1, -1, 2, 1)
        tmp_61 = None
        tmp_63 = tmp_62.transpose(1, 2)
        tmp_62 = None
        tmp_64 = torch.nn.functional.linear(tmp_50, tmp_17, tmp_16)
        tmp_17 = tmp_16 = None
        tmp_65 = tmp_64.view(1, -1, 2, 1)
        tmp_64 = None
        tmp_66 = tmp_65.transpose(1, 2)
        tmp_65 = None
        tmp_67 = torch.nn.functional.scaled_dot_product_attention(tmp_60, tmp_63, tmp_66, attn_mask=tmp_57, dropout_p=0.0, is_causal=False)
        tmp_60 = tmp_63 = tmp_66 = None
        tmp_68 = tmp_67.transpose(1, 2)
        tmp_67 = None
        tmp_69 = tmp_68.reshape(1, 512, 2)
        tmp_68 = None
        tmp_70 = torch.nn.functional.linear(tmp_69, tmp_11, tmp_10)
        tmp_69 = tmp_11 = tmp_10 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.1, False, False)
        tmp_70 = None
        tmp_72 = tmp_71 + tmp_50
        tmp_71 = tmp_50 = None
        tmp_73 = torch.nn.functional.layer_norm(tmp_72, (2,), tmp_9, tmp_8, 1e-12)
        tmp_72 = tmp_9 = tmp_8 = None
        tmp_74 = torch.nn.functional.linear(tmp_73, tmp_19, tmp_18)
        tmp_19 = tmp_18 = None
        tmp_75 = torch.nn.functional.gelu(tmp_74)
        tmp_74 = None
        tmp_76 = torch.nn.functional.linear(tmp_75, tmp_23, tmp_22)
        tmp_75 = tmp_23 = tmp_22 = None
        tmp_77 = torch.nn.functional.dropout(tmp_76, 0.1, False, False)
        tmp_76 = None
        tmp_78 = tmp_77 + tmp_73
        tmp_77 = tmp_73 = None
        tmp_79 = torch.nn.functional.layer_norm(tmp_78, (2,), tmp_21, tmp_20, 1e-12)
        tmp_78 = tmp_21 = tmp_20 = None
        tmp_80 = torch.nn.functional.linear(tmp_79, tmp_31, tmp_30)
        tmp_31 = tmp_30 = None
        tmp_81 = tmp_80.view(1, -1, 2, 1)
        tmp_80 = None
        tmp_82 = tmp_81.transpose(1, 2)
        tmp_81 = None
        tmp_83 = torch.nn.functional.linear(tmp_79, tmp_29, tmp_28)
        tmp_29 = tmp_28 = None
        tmp_84 = tmp_83.view(1, -1, 2, 1)
        tmp_83 = None
        tmp_85 = tmp_84.transpose(1, 2)
        tmp_84 = None
        tmp_86 = torch.nn.functional.linear(tmp_79, tmp_33, tmp_32)
        tmp_33 = tmp_32 = None
        tmp_87 = tmp_86.view(1, -1, 2, 1)
        tmp_86 = None
        tmp_88 = tmp_87.transpose(1, 2)
        tmp_87 = None
        tmp_89 = torch.nn.functional.scaled_dot_product_attention(tmp_82, tmp_85, tmp_88, attn_mask=tmp_57, dropout_p=0.0, is_causal=False)
        tmp_82 = tmp_85 = tmp_88 = tmp_57 = None
        tmp_90 = tmp_89.transpose(1, 2)
        tmp_89 = None
        tmp_91 = tmp_90.reshape(1, 512, 2)
        tmp_90 = None
        tmp_92 = torch.nn.functional.linear(tmp_91, tmp_27, tmp_26)
        tmp_91 = tmp_27 = tmp_26 = None
        tmp_93 = torch.nn.functional.dropout(tmp_92, 0.1, False, False)
        tmp_92 = None
        tmp_94 = tmp_93 + tmp_79
        tmp_93 = tmp_79 = None
        tmp_95 = torch.nn.functional.layer_norm(tmp_94, (2,), tmp_25, tmp_24, 1e-12)
        tmp_94 = tmp_25 = tmp_24 = None
        tmp_96 = torch.nn.functional.linear(tmp_95, tmp_35, tmp_34)
        tmp_35 = tmp_34 = None
        tmp_97 = torch.nn.functional.gelu(tmp_96)
        tmp_96 = None
        tmp_98 = torch.nn.functional.linear(tmp_97, tmp_39, tmp_38)
        tmp_97 = tmp_39 = tmp_38 = None
        tmp_99 = torch.nn.functional.dropout(tmp_98, 0.1, False, False)
        tmp_98 = None
        tmp_100 = tmp_99 + tmp_95
        tmp_99 = tmp_95 = None
        tmp_101 = torch.nn.functional.layer_norm(tmp_100, (2,), tmp_37, tmp_36, 1e-12)
        tmp_100 = tmp_37 = tmp_36 = None
        tmp_102 = tmp_101[slice(None, None, None), 0]
        tmp_103 = torch.nn.functional.linear(tmp_102, tmp_41, tmp_40)
        tmp_102 = tmp_41 = tmp_40 = None
        tmp_104 = torch.tanh(tmp_103)
        tmp_103 = None
        return (tmp_101, tmp_104)