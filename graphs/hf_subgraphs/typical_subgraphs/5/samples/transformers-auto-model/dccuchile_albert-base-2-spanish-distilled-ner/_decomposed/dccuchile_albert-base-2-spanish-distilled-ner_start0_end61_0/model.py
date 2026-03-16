import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28):
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
        tmp_29 = tmp_2[slice(None, None, None), slice(0, 512, None)]
        tmp_2 = None
        tmp_30 = torch.nn.functional.embedding(tmp_1, tmp_7, 0, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_31 = torch.nn.functional.embedding(tmp_28, tmp_6, None, None, 2.0, False, False)
        tmp_28 = tmp_6 = None
        tmp_32 = tmp_30 + tmp_31
        tmp_30 = tmp_31 = None
        tmp_33 = torch.nn.functional.embedding(tmp_29, tmp_5, None, None, 2.0, False, False)
        tmp_29 = tmp_5 = None
        tmp_32 += tmp_33
        tmp_34 = tmp_32
        tmp_32 = tmp_33 = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (128,), tmp_4, tmp_3, 1e-12)
        tmp_34 = tmp_4 = tmp_3 = None
        tmp_36 = torch.nn.functional.dropout(tmp_35, 0, False, False)
        tmp_35 = None
        tmp_37 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_38 = tmp_37.expand(4, 1, 512, 512)
        tmp_37 = None
        tmp_39 = tmp_38.to(torch.float32)
        tmp_38 = None
        tmp_40 = torch.tensor(1.0, dtype=torch.float32)
        tmp_41 = tmp_40 - tmp_39
        tmp_40 = tmp_39 = None
        tmp_42 = tmp_41.to(torch.bool)
        tmp_43 = tmp_41.masked_fill(tmp_42, -3.4028234663852886e+38)
        tmp_41 = tmp_42 = None
        tmp_44 = torch.nn.functional.linear(tmp_36, tmp_25, tmp_24)
        tmp_36 = tmp_25 = tmp_24 = None
        tmp_45 = torch.nn.functional.linear(tmp_44, tmp_15, tmp_14)
        tmp_46 = tmp_45.view(4, -1, 12, 64)
        tmp_45 = None
        tmp_47 = tmp_46.transpose(1, 2)
        tmp_46 = None
        tmp_48 = torch.nn.functional.linear(tmp_44, tmp_13, tmp_12)
        tmp_49 = tmp_48.view(4, -1, 12, 64)
        tmp_48 = None
        tmp_50 = tmp_49.transpose(1, 2)
        tmp_49 = None
        tmp_51 = torch.nn.functional.linear(tmp_44, tmp_17, tmp_16)
        tmp_52 = tmp_51.view(4, -1, 12, 64)
        tmp_51 = None
        tmp_53 = tmp_52.transpose(1, 2)
        tmp_52 = None
        tmp_54 = torch.nn.functional.scaled_dot_product_attention(query=tmp_47, key=tmp_50, value=tmp_53, attn_mask=tmp_43, dropout_p=0.0, is_causal=False)
        tmp_47 = tmp_50 = tmp_53 = None
        tmp_55 = tmp_54.transpose(1, 2)
        tmp_54 = None
        tmp_56 = tmp_55.reshape(4, 512, 768)
        tmp_55 = None
        tmp_57 = torch.nn.functional.linear(tmp_56, tmp_11, tmp_10)
        tmp_56 = None
        tmp_58 = torch.nn.functional.dropout(tmp_57, 0, False, False)
        tmp_57 = None
        tmp_59 = tmp_44 + tmp_58
        tmp_44 = tmp_58 = None
        tmp_60 = torch.nn.functional.layer_norm(tmp_59, (768,), tmp_9, tmp_8, 1e-12)
        tmp_59 = None
        tmp_61 = torch.nn.functional.linear(tmp_60, tmp_21, tmp_20)
        tmp_62 = torch.nn.functional.gelu(tmp_61)
        tmp_61 = None
        tmp_63 = torch.nn.functional.linear(tmp_62, tmp_19, tmp_18)
        tmp_62 = None
        tmp_64 = tmp_63 + tmp_60
        tmp_63 = tmp_60 = None
        tmp_65 = torch.nn.functional.layer_norm(tmp_64, (768,), tmp_23, tmp_22, 1e-12)
        tmp_64 = None
        tmp_66 = torch.nn.functional.linear(tmp_65, tmp_15, tmp_14)
        tmp_15 = tmp_14 = None
        tmp_67 = tmp_66.view(4, -1, 12, 64)
        tmp_66 = None
        tmp_68 = tmp_67.transpose(1, 2)
        tmp_67 = None
        tmp_69 = torch.nn.functional.linear(tmp_65, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_70 = tmp_69.view(4, -1, 12, 64)
        tmp_69 = None
        tmp_71 = tmp_70.transpose(1, 2)
        tmp_70 = None
        tmp_72 = torch.nn.functional.linear(tmp_65, tmp_17, tmp_16)
        tmp_17 = tmp_16 = None
        tmp_73 = tmp_72.view(4, -1, 12, 64)
        tmp_72 = None
        tmp_74 = tmp_73.transpose(1, 2)
        tmp_73 = None
        tmp_75 = torch.nn.functional.scaled_dot_product_attention(query=tmp_68, key=tmp_71, value=tmp_74, attn_mask=tmp_43, dropout_p=0.0, is_causal=False)
        tmp_68 = tmp_71 = tmp_74 = tmp_43 = None
        tmp_76 = tmp_75.transpose(1, 2)
        tmp_75 = None
        tmp_77 = tmp_76.reshape(4, 512, 768)
        tmp_76 = None
        tmp_78 = torch.nn.functional.linear(tmp_77, tmp_11, tmp_10)
        tmp_77 = tmp_11 = tmp_10 = None
        tmp_79 = torch.nn.functional.dropout(tmp_78, 0, False, False)
        tmp_78 = None
        tmp_80 = tmp_65 + tmp_79
        tmp_65 = tmp_79 = None
        tmp_81 = torch.nn.functional.layer_norm(tmp_80, (768,), tmp_9, tmp_8, 1e-12)
        tmp_80 = tmp_9 = tmp_8 = None
        tmp_82 = torch.nn.functional.linear(tmp_81, tmp_21, tmp_20)
        tmp_21 = tmp_20 = None
        tmp_83 = torch.nn.functional.gelu(tmp_82)
        tmp_82 = None
        tmp_84 = torch.nn.functional.linear(tmp_83, tmp_19, tmp_18)
        tmp_83 = tmp_19 = tmp_18 = None
        tmp_85 = tmp_84 + tmp_81
        tmp_84 = tmp_81 = None
        tmp_86 = torch.nn.functional.layer_norm(tmp_85, (768,), tmp_23, tmp_22, 1e-12)
        tmp_85 = tmp_23 = tmp_22 = None
        tmp_87 = tmp_86[slice(None, None, None), 0]
        tmp_88 = torch.nn.functional.linear(tmp_87, tmp_27, tmp_26)
        tmp_87 = tmp_27 = tmp_26 = None
        tmp_89 = torch.tanh(tmp_88)
        tmp_88 = None
        return (tmp_86, tmp_89)