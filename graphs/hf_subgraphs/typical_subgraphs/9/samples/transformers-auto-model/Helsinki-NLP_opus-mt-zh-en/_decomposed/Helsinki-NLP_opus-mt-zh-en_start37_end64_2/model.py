import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_1, in_2, in_3):
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
        tmp_17 = torch.nn.functional.linear(in_1, tmp_16, tmp_15)
        tmp_16 = tmp_15 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, p=0.1, training=False)
        tmp_17 = None
        tmp_19 = in_3 + tmp_18
        tmp_18 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (512,), tmp_14, tmp_13, 1e-05)
        tmp_19 = tmp_14 = tmp_13 = None
        tmp_21 = torch.nn.functional.linear(tmp_20, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_22 = tmp_21.view(1, 26, -1, 64)
        tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = torch.nn.functional.linear(tmp_0, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_25 = torch.nn.functional.linear(tmp_0, tmp_10, tmp_9)
        tmp_0 = tmp_10 = tmp_9 = None
        tmp_26 = tmp_24.view(1, 26, -1, 64)
        tmp_24 = None
        tmp_27 = tmp_26.transpose(1, 2)
        tmp_26 = None
        tmp_28 = tmp_25.view(1, 26, -1, 64)
        tmp_25 = None
        tmp_29 = tmp_28.transpose(1, 2)
        tmp_28 = None
        tmp_30 = in_2[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 26, None)]
        tmp_31 = tmp_23.contiguous()
        tmp_23 = None
        tmp_32 = tmp_27.contiguous()
        tmp_33 = tmp_29.contiguous()
        tmp_34 = torch.nn.functional.scaled_dot_product_attention(tmp_31, tmp_32, tmp_33, attn_mask=tmp_30, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_31 = tmp_32 = tmp_33 = tmp_30 = None
        tmp_35 = tmp_34.transpose(1, 2)
        tmp_34 = None
        tmp_36 = tmp_35.contiguous()
        tmp_35 = None
        tmp_37 = tmp_36.reshape(1, 26, -1)
        tmp_36 = None
        tmp_38 = tmp_37.contiguous()
        tmp_37 = None
        tmp_39 = torch.nn.functional.linear(tmp_38, tmp_6, tmp_5)
        tmp_38 = tmp_6 = tmp_5 = None
        tmp_40 = torch.nn.functional.dropout(tmp_39, p=0.1, training=False)
        tmp_39 = None
        tmp_41 = tmp_20 + tmp_40
        tmp_20 = tmp_40 = None
        tmp_42 = torch.nn.functional.layer_norm(tmp_41, (512,), tmp_2, tmp_1, 1e-05)
        tmp_41 = tmp_2 = tmp_1 = None
        tmp_43 = torch.nn.functional.linear(tmp_42, tmp_12, tmp_11)
        tmp_12 = tmp_11 = None
        return (tmp_42, tmp_27, tmp_43, tmp_29)