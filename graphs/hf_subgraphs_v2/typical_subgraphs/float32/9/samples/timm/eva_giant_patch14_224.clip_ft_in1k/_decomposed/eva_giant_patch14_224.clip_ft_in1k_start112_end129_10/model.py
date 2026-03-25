import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = torch.nn.functional.gelu(in_1, approximate='none')
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_1, tmp_0)
        tmp_9 = tmp_1 = tmp_0 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = in_0 + tmp_11
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (1408,), tmp_7, tmp_6, 1e-06)
        tmp_7 = tmp_6 = None
        tmp_14 = torch.cat((tmp_4, tmp_2, tmp_5))
        tmp_4 = tmp_2 = tmp_5 = None
        tmp_15 = torch.nn.functional.linear(tmp_13, weight=tmp_3, bias=tmp_14)
        tmp_13 = tmp_3 = tmp_14 = None
        tmp_16 = tmp_15.reshape(1, 257, 3, 16, -1)
        tmp_15 = None
        tmp_17 = tmp_16.permute(2, 0, 3, 1, 4)
        tmp_16 = None
        tmp_18 = tmp_17.unbind(0)
        tmp_17 = None
        tmp_19 = tmp_18[0]
        tmp_20 = tmp_18[1]
        tmp_21 = tmp_18[2]
        tmp_18 = None
        tmp_22 = torch.nn.functional.scaled_dot_product_attention(tmp_19, tmp_20, tmp_21, attn_mask=None, dropout_p=0.0)
        tmp_19 = tmp_20 = tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = tmp_23.reshape(1, 257, 1408)
        tmp_23 = None
        return (tmp_12, tmp_24)