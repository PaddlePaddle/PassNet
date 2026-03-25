import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_2.view(1, 22, -1, 4)
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_7 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_8 = tmp_6.view(1, 22, -1, 4)
        tmp_6 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = tmp_7.view(1, 22, -1, 4)
        tmp_7 = None
        tmp_11 = tmp_10.transpose(1, 2)
        tmp_10 = None
        tmp_12 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 22, None)]
        tmp_13 = tmp_5.contiguous()
        tmp_5 = None
        tmp_14 = tmp_9.contiguous()
        tmp_15 = tmp_11.contiguous()
        tmp_16 = torch.nn.functional.scaled_dot_product_attention(tmp_13, tmp_14, tmp_15, attn_mask=tmp_12, dropout_p=0.0, scale=0.5, is_causal=False)
        tmp_13 = tmp_14 = tmp_15 = tmp_12 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = tmp_17.contiguous()
        tmp_17 = None
        tmp_19 = tmp_18.reshape(1, 22, -1)
        tmp_18 = None
        tmp_20 = tmp_19.contiguous()
        tmp_19 = None
        return (tmp_20, tmp_9, tmp_11)