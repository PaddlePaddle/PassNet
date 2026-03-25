import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34):
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
        tmp_33 = torch.nn.functional.linear(in_33, tmp_19, tmp_18)
        tmp_19 = tmp_18 = None
        tmp_34 = in_34 + tmp_33
        tmp_33 = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (1152,), tmp_13, tmp_12, 1e-06)
        tmp_13 = tmp_12 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_15, tmp_14)
        tmp_35 = tmp_15 = tmp_14 = None
        tmp_37 = torch.nn.functional.gelu(tmp_36, approximate='tanh')
        tmp_36 = None
        tmp_38 = torch.nn.functional.linear(tmp_37, tmp_17, tmp_16)
        tmp_37 = tmp_17 = tmp_16 = None
        tmp_39 = tmp_34 + tmp_38
        tmp_34 = tmp_38 = None
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (1152,), tmp_32, tmp_31, 1e-06)
        tmp_39 = tmp_32 = tmp_31 = None
        tmp_41 = tmp_30.repeat(1, 1, 1)
        tmp_30 = None
        tmp_42 = tmp_41.transpose(1, 0)
        tmp_41 = None
        tmp_43 = tmp_40.transpose(1, 0)
        tmp_44 = torch.nn.functional.multi_head_attention_forward(tmp_42, tmp_43, tmp_43, 1152, 16, tmp_23, tmp_22, None, None, False, 0.0, tmp_21, tmp_20, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_42 = tmp_43 = tmp_23 = tmp_22 = tmp_21 = tmp_20 = None
        tmp_45 = tmp_44[0]
        tmp_44 = None
        tmp_46 = tmp_45.transpose(1, 0)
        tmp_45 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (1152,), tmp_25, tmp_24, 1e-06)
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
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (1152,), tmp_5, tmp_4, 1e-06)
        tmp_5 = tmp_4 = None
        tmp_59 = torch.nn.functional.linear(tmp_58, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_60 = torch.nn.functional.linear(tmp_58, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_61 = torch.nn.functional.linear(tmp_58, tmp_11, tmp_10)
        tmp_58 = tmp_11 = tmp_10 = None
        return (tmp_57, tmp_60, tmp_40, tmp_52, tmp_59, tmp_61)