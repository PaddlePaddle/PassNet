import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = w_10
        tmp_12 = w_11
        tmp_13 = w_12
        tmp_14 = w_13
        tmp_15 = w_14
        tmp_16 = w_15
        tmp_17 = w_16
        tmp_18 = w_17
        tmp_19 = w_18
        tmp_20 = w_19
        tmp_21 = w_20
        tmp_22 = w_21
        tmp_23 = w_22
        tmp_24 = w_23
        tmp_25 = w_24
        tmp_26 = w_25
        tmp_27 = w_26
        tmp_28 = w_27
        tmp_29 = w_28
        tmp_30 = w_29
        tmp_31 = w_30
        tmp_32 = w_31
        tmp_33 = torch.nn.functional.linear(in_1, tmp_19, tmp_18)
        tmp_19 = tmp_18 = None
        tmp_34 = in_2 + tmp_33
        tmp_33 = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (768,), tmp_13, tmp_12, 1e-06)
        tmp_13 = tmp_12 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_15, tmp_14)
        tmp_35 = tmp_15 = tmp_14 = None
        tmp_37 = torch.nn.functional.gelu(tmp_36, approximate='tanh')
        tmp_36 = None
        tmp_38 = torch.nn.functional.linear(tmp_37, tmp_17, tmp_16)
        tmp_37 = tmp_17 = tmp_16 = None
        tmp_39 = tmp_34 + tmp_38
        tmp_34 = tmp_38 = None
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (768,), tmp_32, tmp_31, 1e-06)
        tmp_39 = tmp_32 = tmp_31 = None
        tmp_41 = tmp_30.repeat(1, 1, 1)
        tmp_30 = None
        tmp_42 = tmp_41.transpose(1, 0)
        tmp_41 = None
        tmp_43 = tmp_40.transpose(1, 0)
        tmp_44 = torch.nn.functional.multi_head_attention_forward(tmp_42, tmp_43, tmp_43, 768, 12, tmp_23, tmp_22, None, None, False, 0.0, tmp_21, tmp_20, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_42 = tmp_43 = tmp_23 = tmp_22 = tmp_21 = tmp_20 = None
        tmp_45 = tmp_44[0]
        tmp_44 = None
        tmp_46 = tmp_45.transpose(1, 0)
        tmp_45 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (768,), tmp_25, tmp_24, 1e-06)
        tmp_25 = tmp_24 = None
        tmp_48 = torch.nn.functional.linear(tmp_47, tmp_27, tmp_26)
        tmp_47 = tmp_27 = tmp_26 = None
        tmp_49 = torch.nn.functional.gelu(tmp_48, approximate='tanh')
        tmp_48 = None
        tmp_50 = torch.nn.functional.linear(tmp_49, tmp_29, tmp_28)
        tmp_49 = tmp_29 = tmp_28 = None
        tmp_51 = tmp_46 + tmp_50
        tmp_46 = tmp_50 = None
        tmp_52 = tmp_51[slice(None, None, None), 0]
        tmp_51 = None
        tmp_53 = tmp_0.view(-1, 7)
        tmp_0 = None
        tmp_54 = tmp_1[slice(None, None, None), slice(None, 7, None)]
        tmp_1 = None
        tmp_55 = torch.nn.functional.embedding(tmp_53, tmp_3, None, None, 2.0, False, False)
        tmp_53 = tmp_3 = None
        tmp_56 = torch.nn.functional.embedding(tmp_54, tmp_2, None, None, 2.0, False, False)
        tmp_54 = tmp_2 = None
        tmp_57 = tmp_55 + tmp_56
        tmp_55 = tmp_56 = None
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (768,), tmp_5, tmp_4, 1e-06)
        tmp_5 = tmp_4 = None
        tmp_59 = torch.nn.functional.linear(tmp_58, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_60 = torch.nn.functional.linear(tmp_58, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_61 = torch.nn.functional.linear(tmp_58, tmp_11, tmp_10)
        tmp_58 = tmp_11 = tmp_10 = None
        return (tmp_57, tmp_60, tmp_40, tmp_52, tmp_59, tmp_61)