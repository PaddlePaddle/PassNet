import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_2 : torch.Tensor):
        tmp_9 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_10 = tmp_9.to(dtype = torch.float32);  tmp_9 = None
        tmp_11 = 1.0 - tmp_10;  tmp_10 = None
        tmp_12 = tmp_11 * -3.4028234663852886e+38;  tmp_11 = None
        tmp_13 = w_0[(slice(None, None, None), slice(0, 11, None))];  w_0 = None
        tmp_14 = torch.nn.functional.embedding(in_1, w_5, 0, None, 2.0, False, False);  in_1 = w_5 = None
        tmp_15 = torch.nn.functional.embedding(in_2, w_4, None, None, 2.0, False, False);  in_2 = w_4 = None
        tmp_16 = tmp_14 + tmp_15;  tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.embedding(tmp_13, w_3, None, None, 2.0, False, False);  tmp_13 = w_3 = None
        tmp_16 += tmp_17;  tmp_18 = tmp_16;  tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.1, False, False);  tmp_18 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (32,), w_2, w_1, 1e-12);  tmp_19 = w_2 = w_1 = None
        return (tmp_20, tmp_12)
        