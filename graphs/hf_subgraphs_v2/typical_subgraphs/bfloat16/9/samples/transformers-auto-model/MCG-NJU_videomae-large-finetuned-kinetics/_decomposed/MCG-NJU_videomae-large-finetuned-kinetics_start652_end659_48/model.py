import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.nn.functional.gelu(in_1);  in_1 = None
        linear = torch.nn.functional.linear(tmp_6, w_5, w_4);  tmp_6 = w_5 = w_4 = None
        tmp_8 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_9 = tmp_8 + in_0;  tmp_8 = in_0 = None
        tmp_10 = tmp_9.mean(1);  tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (1024,), w_3, w_2, 1e-05);  tmp_10 = w_3 = w_2 = None
        linear_1 = torch.nn.functional.linear(tmp_11, w_1, w_0);  tmp_11 = w_1 = w_0 = None
        return (linear_1,)
        