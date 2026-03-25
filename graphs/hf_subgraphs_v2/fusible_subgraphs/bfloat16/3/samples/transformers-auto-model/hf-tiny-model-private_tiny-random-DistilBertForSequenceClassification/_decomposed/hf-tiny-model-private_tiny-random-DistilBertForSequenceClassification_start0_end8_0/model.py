import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_7 = torch.nn.functional.embedding(in_1, in_6, 0, None, 2.0, False, False);  in_1 = in_6 = None
        tmp_8 = in_2[(slice(None, None, None), slice(None, 64, None))];  in_2 = None
        tmp_9 = torch.nn.functional.embedding(tmp_8, in_5, None, None, 2.0, False, False);  tmp_8 = in_5 = None
        tmp_10 = tmp_7 + tmp_9;  tmp_7 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (32,), in_4, in_3, 1e-12);  tmp_10 = in_4 = in_3 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False);  tmp_11 = None
        tmp_13 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_14 = tmp_13.expand(32, 1, 64, 64);  tmp_13 = None
        return (tmp_12, tmp_14)
        