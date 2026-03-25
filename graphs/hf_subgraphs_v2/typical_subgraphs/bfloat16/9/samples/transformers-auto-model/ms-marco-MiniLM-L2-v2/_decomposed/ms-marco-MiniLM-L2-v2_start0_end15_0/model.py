import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_2 : torch.Tensor):
        tmp_9 = w_0[(slice(None, None, None), slice(0, 34, None))];  w_0 = None
        tmp_10 = torch.nn.functional.embedding(in_1, w_5, 0, None, 2.0, False, False);  in_1 = w_5 = None
        tmp_11 = torch.nn.functional.embedding(in_2, w_4, None, None, 2.0, False, False);  in_2 = w_4 = None
        tmp_12 = tmp_10 + tmp_11;  tmp_10 = tmp_11 = None
        tmp_13 = torch.nn.functional.embedding(tmp_9, w_3, None, None, 2.0, False, False);  tmp_9 = w_3 = None
        tmp_12 += tmp_13;  tmp_14 = tmp_12;  tmp_12 = tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (384,), w_2, w_1, 1e-12);  tmp_14 = w_2 = w_1 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.1, False, False);  tmp_15 = None
        tmp_17 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_18 = tmp_17.expand(2, 1, 34, 34);  tmp_17 = None
        tmp_19 = tmp_18.to(torch.float32);  tmp_18 = None
        tmp_20 = torch.tensor(1.0, dtype = torch.float32)
        tmp_21 = tmp_20 - tmp_19;  tmp_20 = tmp_19 = None
        tmp_22 = tmp_21.to(torch.bool)
        tmp_23 = tmp_21.masked_fill(tmp_22, -3.4028234663852886e+38);  tmp_21 = tmp_22 = None
        return (tmp_16, tmp_23)
        