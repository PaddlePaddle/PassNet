import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, in_0, in_1):
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
        tmp_18 = torch.nn.functional.gelu(in_0, approximate='none')
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.0, False, False)
        tmp_18 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_1, tmp_0)
        tmp_19 = tmp_1 = tmp_0 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.0, False, False)
        tmp_20 = None
        tmp_22 = in_1 + tmp_21
        tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = tmp_23.view(1, 768, 16, 16)
        tmp_23 = None
        tmp_25 = torch.conv2d(tmp_24, tmp_17, tmp_16, (1, 1), (1, 1), (1, 1), 768)
        tmp_17 = tmp_16 = None
        tmp_26 = tmp_25 + tmp_24
        tmp_25 = tmp_24 = None
        tmp_27 = tmp_26.flatten(2)
        tmp_26 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = torch.nn.functional.layer_norm(tmp_28, (768,), tmp_11, tmp_10, 1e-05)
        tmp_11 = tmp_10 = None
        tmp_30 = tmp_29.transpose(0, 1)
        tmp_31 = tmp_29.transpose(0, 1)
        tmp_29 = None
        tmp_32 = torch.nn.functional.multi_head_attention_forward(tmp_30, tmp_31, tmp_31, 768, 24, tmp_5, tmp_4, None, None, False, 0.0, tmp_3, tmp_2, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_30 = tmp_31 = tmp_5 = tmp_4 = tmp_3 = tmp_2 = None
        tmp_33 = tmp_32[0]
        tmp_32 = None
        tmp_34 = tmp_33.transpose(0, 1)
        tmp_33 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.0, False, False)
        tmp_34 = None
        tmp_36 = 0.0 + tmp_35
        tmp_35 = None
        tmp_37 = tmp_28 + tmp_36
        tmp_28 = tmp_36 = None
        tmp_38 = torch.nn.functional.layer_norm(tmp_37, (768,), tmp_13, tmp_12, 1e-05)
        tmp_13 = tmp_12 = None
        tmp_39 = torch.nn.functional.linear(tmp_38, tmp_7, tmp_6)
        tmp_38 = tmp_7 = tmp_6 = None
        tmp_40 = torch.nn.functional.gelu(tmp_39, approximate='none')
        tmp_39 = None
        tmp_41 = torch.nn.functional.dropout(tmp_40, 0.0, False, False)
        tmp_40 = None
        tmp_42 = torch.nn.functional.linear(tmp_41, tmp_9, tmp_8)
        tmp_41 = tmp_9 = tmp_8 = None
        tmp_43 = torch.nn.functional.dropout(tmp_42, 0.0, False, False)
        tmp_42 = None
        tmp_44 = tmp_37 + tmp_43
        tmp_37 = tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (768,), tmp_15, tmp_14, 1e-05)
        tmp_44 = tmp_15 = tmp_14 = None
        tmp_46 = tmp_45.reshape(1, 16, 16, -1)
        tmp_45 = None
        tmp_47 = tmp_46.permute(0, 3, 1, 2)
        tmp_46 = None
        tmp_48 = tmp_47.contiguous()
        tmp_47 = None
        return (tmp_48,)