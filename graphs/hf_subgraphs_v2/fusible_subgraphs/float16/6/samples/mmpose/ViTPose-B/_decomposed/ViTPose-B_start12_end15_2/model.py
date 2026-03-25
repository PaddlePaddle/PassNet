import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_1, in_0, in_2, dropout_p = 0.0);  in_1 = in_0 = in_2 = None
        tmp_1 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_2 = tmp_1.reshape(16, 192, 768);  tmp_1 = None
        return (tmp_2,)
        