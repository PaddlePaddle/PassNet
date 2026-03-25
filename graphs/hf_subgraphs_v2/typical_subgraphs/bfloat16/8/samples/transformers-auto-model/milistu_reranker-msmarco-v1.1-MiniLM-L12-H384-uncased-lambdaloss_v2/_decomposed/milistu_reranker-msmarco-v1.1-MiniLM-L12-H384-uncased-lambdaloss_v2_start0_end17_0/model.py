import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        tmp_11 = in_2[(slice(None, None, None), slice(0, 64, None))];  in_2 = None
        tmp_12 = torch.nn.functional.embedding(in_1, in_7, 0, None, 2.0, False, False);  in_1 = in_7 = None
        tmp_13 = torch.nn.functional.embedding(in_10, in_6, None, None, 2.0, False, False);  in_10 = in_6 = None
        tmp_14 = tmp_12 + tmp_13;  tmp_12 = tmp_13 = None
        tmp_15 = torch.nn.functional.embedding(tmp_11, in_5, None, None, 2.0, False, False);  tmp_11 = in_5 = None
        tmp_14 += tmp_15;  tmp_16 = tmp_14;  tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (384,), in_4, in_3, 1e-12);  tmp_16 = in_4 = in_3 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False);  tmp_17 = None
        tmp_19 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_20 = tmp_19.expand(128, 1, 64, 64);  tmp_19 = None
        tmp_21 = tmp_20.to(torch.float32);  tmp_20 = None
        tmp_22 = torch.tensor(1.0, dtype = torch.float32)
        tmp_23 = tmp_22 - tmp_21;  tmp_22 = tmp_21 = None
        tmp_24 = tmp_23.to(torch.bool)
        tmp_25 = tmp_23.masked_fill(tmp_24, -3.4028234663852886e+38);  tmp_23 = tmp_24 = None
        linear = torch.nn.functional.linear(tmp_18, in_9, in_8);  in_9 = in_8 = None
        tmp_27 = linear.view(128, -1, 12, 32);  linear = None
        return (tmp_18, tmp_25, tmp_27)
        