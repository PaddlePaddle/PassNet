import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.scaled_dot_product_attention(in_0, in_1, in_2, attn_mask=None, dropout_p=0.0, scale=0.10206207261596575, is_causal=False)
        tmp_1 = tmp_0.transpose(1, 2)
        tmp_0 = None
        tmp_2 = tmp_1.contiguous()
        tmp_1 = None
        tmp_3 = tmp_2.reshape(1, 576, 1536)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        return (tmp_4,)