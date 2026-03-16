import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0, in_1, in_2):
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
        tmp_16 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, p=0.1, training=False)
        tmp_16 = None
        tmp_18 = in_2 + tmp_17
        tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (1024,), tmp_3, tmp_2, 1e-05)
        tmp_18 = tmp_3 = tmp_2 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_21 = tmp_20.view(1, 1, -1, 64)
        tmp_20 = None
        tmp_22 = tmp_21.transpose(1, 2)
        tmp_21 = None
        tmp_23 = torch.nn.functional.linear(tmp_19, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_24 = torch.nn.functional.linear(tmp_19, tmp_15, tmp_14)
        tmp_15 = tmp_14 = None
        tmp_25 = tmp_23.view(1, 1, -1, 64)
        tmp_23 = None
        tmp_26 = tmp_25.transpose(1, 2)
        tmp_25 = None
        tmp_27 = tmp_24.view(1, 1, -1, 64)
        tmp_24 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 1, None)]
        tmp_30 = tmp_22.contiguous()
        tmp_22 = None
        tmp_31 = tmp_26.contiguous()
        tmp_26 = None
        tmp_32 = tmp_28.contiguous()
        tmp_28 = None
        tmp_33 = torch.nn.functional.scaled_dot_product_attention(tmp_30, tmp_31, tmp_32, attn_mask=tmp_29, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_30 = tmp_31 = tmp_32 = tmp_29 = None
        tmp_34 = tmp_33.transpose(1, 2)
        tmp_33 = None
        tmp_35 = tmp_34.contiguous()
        tmp_34 = None
        tmp_36 = tmp_35.reshape(1, 1, -1)
        tmp_35 = None
        tmp_37 = tmp_36.contiguous()
        tmp_36 = None
        tmp_38 = torch.nn.functional.linear(tmp_37, tmp_11, tmp_10)
        tmp_37 = tmp_11 = tmp_10 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, p=0.1, training=False)
        tmp_38 = None
        tmp_40 = tmp_19 + tmp_39
        tmp_19 = tmp_39 = None
        tmp_41 = torch.nn.functional.layer_norm(tmp_40, (1024,), tmp_7, tmp_6, 1e-05)
        tmp_40 = tmp_7 = tmp_6 = None
        tmp_42 = torch.nn.functional.linear(tmp_41, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        return (tmp_41, tmp_42)