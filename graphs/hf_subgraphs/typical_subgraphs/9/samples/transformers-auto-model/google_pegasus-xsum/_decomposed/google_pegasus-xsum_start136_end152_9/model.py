import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, in_1):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = in_1.view(1, 10, -1, 64)
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = torch.nn.functional.linear(tmp_0, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_8 = torch.nn.functional.linear(tmp_0, tmp_4, tmp_3)
        tmp_0 = tmp_4 = tmp_3 = None
        tmp_9 = tmp_7.view(1, 10, -1, 64)
        tmp_7 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_8.view(1, 10, -1, 64)
        tmp_8 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = tmp_6.contiguous()
        tmp_6 = None
        tmp_14 = tmp_10.contiguous()
        tmp_15 = tmp_12.contiguous()
        tmp_16 = torch.nn.functional.scaled_dot_product_attention(tmp_13, tmp_14, tmp_15, attn_mask=None, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_13 = tmp_14 = tmp_15 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = tmp_17.contiguous()
        tmp_17 = None
        tmp_19 = tmp_18.reshape(1, 10, -1)
        tmp_18 = None
        tmp_20 = tmp_19.contiguous()
        tmp_19 = None
        return (tmp_20, tmp_10, tmp_12)