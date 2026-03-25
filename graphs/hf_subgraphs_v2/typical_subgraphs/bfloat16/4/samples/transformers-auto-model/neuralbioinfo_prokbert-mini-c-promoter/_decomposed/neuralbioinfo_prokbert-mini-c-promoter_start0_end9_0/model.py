import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_7 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_8 = tmp_7.to(dtype = torch.float32);  tmp_7 = None
        tmp_9 = 1.0 - tmp_8;  tmp_8 = None
        tmp_10 = tmp_9 * -3.4028234663852886e+38;  tmp_9 = None
        tmp_11 = torch.nn.functional.embedding(in_1, in_3, 0, None, 2.0, False, False);  in_1 = in_3 = None
        tmp_12 = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False);  in_6 = in_2 = None
        tmp_13 = tmp_11 + tmp_12;  tmp_11 = tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.1, False, False);  tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (384,), in_5, in_4, 1e-12);  in_5 = in_4 = None
        return (tmp_14, tmp_10, tmp_15)
        