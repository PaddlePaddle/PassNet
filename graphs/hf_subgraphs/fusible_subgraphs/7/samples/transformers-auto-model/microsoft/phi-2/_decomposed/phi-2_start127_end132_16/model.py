import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.scaled_dot_product_attention(in_2, in_1, in_3, attn_mask=in_0, dropout_p=0.0, scale=0.11180339887498948, is_causal=False)
        tmp_1 = tmp_0.transpose(1, 2)
        tmp_0 = None
        tmp_2 = tmp_1.contiguous()
        tmp_1 = None
        tmp_3 = tmp_2.reshape(1, 256, -1)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        return (tmp_4,)