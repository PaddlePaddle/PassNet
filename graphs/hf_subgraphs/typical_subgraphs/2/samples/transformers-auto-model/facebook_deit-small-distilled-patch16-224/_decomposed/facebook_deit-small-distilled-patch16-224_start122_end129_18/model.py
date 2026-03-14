import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_1.contiguous()
        tmp_1 = in_0.contiguous()
        tmp_2 = in_2.contiguous()
        tmp_3 = torch.nn.functional.scaled_dot_product_attention(tmp_0, tmp_1, tmp_2, attn_mask=None, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_0 = tmp_1 = tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_4.contiguous()
        tmp_4 = None
        tmp_6 = tmp_5.reshape((1, 198, 384))
        tmp_5 = None
        return (tmp_6,)