import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = torch.nn.functional.gelu(in_2)
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_1, tmp_0)
        tmp_10 = tmp_1 = tmp_0 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
        tmp_11 = None
        tmp_13 = in_1 + tmp_12
        tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), tmp_8, tmp_7, 1e-05)
        tmp_8 = tmp_7 = None
        tmp_15 = tmp_14.view(1, 199, 16, -1)
        tmp_16 = tmp_15.permute(0, 2, 1, 3)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_5, tmp_4)
        tmp_16 = tmp_5 = tmp_4 = None
        tmp_18 = tmp_17.view(1, 16, 199, 2, 4)
        tmp_17 = None
        tmp_19 = tmp_18.sum(-1, keepdim=False)
        tmp_18 = None
        tmp_20 = torch.sigmoid(tmp_19)
        tmp_19 = None
        tmp_21 = tmp_20.chunk(2, dim=-1)
        tmp_20 = None
        tmp_22 = tmp_21[0]
        tmp_23 = tmp_21[1]
        tmp_21 = None
        tmp_24 = tmp_23 * tmp_6
        tmp_23 = tmp_6 = None
        tmp_25 = tmp_24 - 1.0
        tmp_24 = None
        tmp_26 = tmp_22 * tmp_25
        tmp_22 = tmp_25 = None
        tmp_27 = tmp_26 + 2.0
        tmp_26 = None
        tmp_28 = tmp_27.view(1, 16, -1, 1)
        tmp_27 = None
        tmp_29 = tmp_28 * in_0
        tmp_28 = None
        tmp_30 = tmp_29.view((1, 16, 199, 199))
        tmp_29 = None
        tmp_31 = torch.nn.functional.linear(tmp_14, tmp_3, tmp_2)
        tmp_14 = tmp_3 = tmp_2 = None
        tmp_32 = tmp_31.chunk(3, -1)
        tmp_31 = None
        tmp_33 = tmp_32[0]
        tmp_34 = tmp_32[1]
        tmp_35 = tmp_32[2]
        tmp_32 = None
        tmp_36 = tmp_33.view((1, 199, 16, 64))
        tmp_33 = None
        tmp_37 = tmp_36.transpose(2, 1)
        tmp_36 = None
        tmp_38 = tmp_34.view((1, 199, 16, 64))
        tmp_34 = None
        tmp_39 = tmp_38.transpose(2, 1)
        tmp_38 = None
        tmp_40 = tmp_35.view((1, 199, 16, 64))
        tmp_35 = None
        tmp_41 = tmp_40.transpose(2, 1)
        tmp_40 = None
        tmp_42 = torch.nn.functional.scaled_dot_product_attention(tmp_37, tmp_39, tmp_41, attn_mask=tmp_30, dropout_p=0.0, is_causal=False)
        tmp_37 = tmp_39 = tmp_41 = tmp_30 = None
        tmp_43 = tmp_42.transpose(1, 2)
        tmp_42 = None
        tmp_44 = tmp_43.reshape(1, -1, 1024)
        tmp_43 = None
        return (tmp_44, tmp_13)