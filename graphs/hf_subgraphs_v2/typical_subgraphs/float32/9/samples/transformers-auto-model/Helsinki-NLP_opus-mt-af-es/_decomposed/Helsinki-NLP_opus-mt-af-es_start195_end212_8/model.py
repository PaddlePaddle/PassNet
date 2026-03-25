import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = in_2.view(1, 20, -1, 64)
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = torch.nn.functional.linear(tmp_0, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_8 = torch.nn.functional.linear(tmp_0, tmp_4, tmp_3)
        tmp_0 = tmp_4 = tmp_3 = None
        tmp_9 = tmp_7.view(1, 20, -1, 64)
        tmp_7 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_8.view(1, 20, -1, 64)
        tmp_8 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = in_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 20, None)]
        tmp_14 = tmp_6.contiguous()
        tmp_6 = None
        tmp_15 = tmp_10.contiguous()
        tmp_16 = tmp_12.contiguous()
        tmp_17 = torch.nn.functional.scaled_dot_product_attention(tmp_14, tmp_15, tmp_16, attn_mask=tmp_13, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_14 = tmp_15 = tmp_16 = tmp_13 = None
        tmp_18 = tmp_17.transpose(1, 2)
        tmp_17 = None
        tmp_19 = tmp_18.contiguous()
        tmp_18 = None
        tmp_20 = tmp_19.reshape(1, 20, -1)
        tmp_19 = None
        tmp_21 = tmp_20.contiguous()
        tmp_20 = None
        return (tmp_21, tmp_10, tmp_12)