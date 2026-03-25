import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = in_0.reshape(1, 197, 3, 16, 64);  in_0 = None
        tmp_3 = tmp_2.permute(2, 0, 3, 1, 4);  tmp_2 = None
        unbind = tmp_3.unbind(0);  tmp_3 = None
        tmp_5 = unbind[0]
        tmp_6 = unbind[1]
        tmp_7 = unbind[2];  unbind = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_5, tmp_6, tmp_7, attn_mask = None, dropout_p = 0.0);  tmp_5 = tmp_6 = tmp_7 = None
        tmp_9 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_10 = tmp_9.reshape(1, 197, 1024);  tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (1024,), w_1, w_0, 1e-05);  tmp_10 = w_1 = w_0 = None
        return (tmp_11,)
        