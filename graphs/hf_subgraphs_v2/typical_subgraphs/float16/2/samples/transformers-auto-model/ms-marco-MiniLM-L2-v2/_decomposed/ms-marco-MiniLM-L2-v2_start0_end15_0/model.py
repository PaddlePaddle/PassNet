import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor):
        tmp_9 = in_2[(slice(None, None, None), slice(0, 128, None))];  in_2 = None
        tmp_10 = torch.nn.functional.embedding(in_1, in_7, 0, None, 2.0, False, False);  in_1 = in_7 = None
        tmp_11 = torch.nn.functional.embedding(in_8, in_6, None, None, 2.0, False, False);  in_8 = in_6 = None
        tmp_12 = tmp_10 + tmp_11;  tmp_10 = tmp_11 = None
        tmp_13 = torch.nn.functional.embedding(tmp_9, in_5, None, None, 2.0, False, False);  tmp_9 = in_5 = None
        tmp_12 += tmp_13;  tmp_14 = tmp_12;  tmp_12 = tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (384,), in_4, in_3, 1e-12);  tmp_14 = in_4 = in_3 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.1, False, False);  tmp_15 = None
        tmp_17 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_18 = tmp_17.expand(16, 1, 128, 128);  tmp_17 = None
        tmp_19 = tmp_18.to(torch.float32);  tmp_18 = None
        tmp_20 = torch.tensor(1.0, dtype = torch.float32)
        tmp_21 = tmp_20 - tmp_19;  tmp_20 = tmp_19 = None
        tmp_22 = tmp_21.to(torch.bool)
        tmp_23 = tmp_21.masked_fill(tmp_22, -3.4028234663852886e+38);  tmp_21 = tmp_22 = None
        return (tmp_16, tmp_23)
        