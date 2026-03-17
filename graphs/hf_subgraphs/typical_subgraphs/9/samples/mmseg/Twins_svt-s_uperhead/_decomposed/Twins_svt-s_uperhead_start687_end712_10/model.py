import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0, in_1):
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
        tmp_16 = torch.nn.functional.gelu(in_0, approximate='none')
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_1, tmp_0)
        tmp_17 = tmp_1 = tmp_0 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.0, False, False)
        tmp_18 = None
        tmp_20 = in_1 + tmp_19
        tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (512,), tmp_11, tmp_10, 1e-05)
        tmp_11 = tmp_10 = None
        tmp_22 = tmp_21.transpose(0, 1)
        tmp_23 = tmp_21.transpose(0, 1)
        tmp_21 = None
        tmp_24 = torch.nn.functional.multi_head_attention_forward(tmp_22, tmp_23, tmp_23, 512, 16, tmp_5, tmp_4, None, None, False, 0.0, tmp_3, tmp_2, training=False, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False)
        tmp_22 = tmp_23 = tmp_5 = tmp_4 = tmp_3 = tmp_2 = None
        tmp_25 = tmp_24[0]
        tmp_24 = None
        tmp_26 = tmp_25.transpose(0, 1)
        tmp_25 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.0, False, False)
        tmp_26 = None
        tmp_28 = 0.0 + tmp_27
        tmp_27 = None
        tmp_29 = tmp_20 + tmp_28
        tmp_20 = tmp_28 = None
        tmp_30 = torch.nn.functional.layer_norm(tmp_29, (512,), tmp_13, tmp_12, 1e-05)
        tmp_13 = tmp_12 = None
        tmp_31 = torch.nn.functional.linear(tmp_30, tmp_7, tmp_6)
        tmp_30 = tmp_7 = tmp_6 = None
        tmp_32 = torch.nn.functional.gelu(tmp_31, approximate='none')
        tmp_31 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.0, False, False)
        tmp_32 = None
        tmp_34 = torch.nn.functional.linear(tmp_33, tmp_9, tmp_8)
        tmp_33 = tmp_9 = tmp_8 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.0, False, False)
        tmp_34 = None
        tmp_36 = tmp_29 + tmp_35
        tmp_29 = tmp_35 = None
        tmp_37 = torch.nn.functional.layer_norm(tmp_36, (512,), tmp_15, tmp_14, 1e-05)
        tmp_36 = tmp_15 = tmp_14 = None
        tmp_38 = tmp_37.reshape(1, 16, 16, -1)
        tmp_37 = None
        tmp_39 = tmp_38.permute(0, 3, 1, 2)
        tmp_38 = None
        tmp_40 = tmp_39.contiguous()
        tmp_39 = None
        return (tmp_40,)