import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor):
        tmp_12 = in_0.view(-1, 64);  in_0 = None
        tmp_13 = w_0[(slice(None, None, None), slice(None, 64, None))];  w_0 = None
        tmp_14 = torch.nn.functional.embedding(tmp_12, w_2, None, None, 2.0, False, False);  tmp_12 = w_2 = None
        tmp_15 = torch.nn.functional.embedding(tmp_13, w_1, None, None, 2.0, False, False);  tmp_13 = w_1 = None
        tmp_16 = tmp_14 + tmp_15;  tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), w_4, w_3, 1e-06);  w_4 = w_3 = None
        linear = torch.nn.functional.linear(tmp_17, w_8, w_7);  w_8 = w_7 = None
        linear_1 = torch.nn.functional.linear(tmp_17, w_6, w_5);  w_6 = w_5 = None
        linear_2 = torch.nn.functional.linear(tmp_17, w_10, w_9);  tmp_17 = w_10 = w_9 = None
        return (tmp_16, linear_1, linear, linear_2)
        