import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = in_1.contiguous();  in_1 = None
        tmp_1 = in_0.contiguous();  in_0 = None
        tmp_2 = in_2.contiguous();  in_2 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_0, tmp_1, tmp_2, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_0 = tmp_1 = tmp_2 = None
        tmp_4 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_5 = tmp_4.contiguous();  tmp_4 = None
        tmp_6 = tmp_5.reshape((1, 197, 768));  tmp_5 = None
        return (tmp_6,)
        