import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.reshape(1, 192, 3, 12, 64);  in_0 = None
        tmp_1 = tmp_0.permute(2, 0, 3, 1, 4);  tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_4 = tmp_1[2];  tmp_1 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_2, tmp_3, tmp_4, dropout_p = 0.0);  tmp_2 = tmp_3 = tmp_4 = None
        tmp_6 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_7 = tmp_6.reshape(1, 192, 768);  tmp_6 = None
        return (tmp_7,)
        