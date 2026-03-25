import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor):
        tmp_7 = torch.nn.functional.embedding(in_1, w_4, 0, None, 2.0, False, False);  in_1 = w_4 = None
        tmp_8 = w_0[(slice(None, None, None), slice(None, 23, None))];  w_0 = None
        tmp_9 = torch.nn.functional.embedding(tmp_8, w_3, None, None, 2.0, False, False);  tmp_8 = w_3 = None
        tmp_10 = tmp_7 + tmp_9;  tmp_7 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (768,), w_2, w_1, 1e-12);  tmp_10 = w_2 = w_1 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False);  tmp_11 = None
        tmp_13 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_14 = tmp_13.expand(1, 1, 23, 23);  tmp_13 = None
        tmp_15 = tmp_14.to(torch.float32);  tmp_14 = None
        tmp_16 = torch.tensor(1.0, dtype = torch.float32)
        tmp_17 = tmp_16 - tmp_15;  tmp_16 = tmp_15 = None
        tmp_18 = tmp_17.to(torch.bool)
        tmp_19 = tmp_17.masked_fill(tmp_18, -3.4028234663852886e+38);  tmp_17 = tmp_18 = None
        return (tmp_19, tmp_12)
        