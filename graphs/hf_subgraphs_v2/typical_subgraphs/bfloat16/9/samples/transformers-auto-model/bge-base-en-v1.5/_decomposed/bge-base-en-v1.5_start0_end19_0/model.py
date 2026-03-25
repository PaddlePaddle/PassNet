import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor):
        tmp_11 = w_1[(slice(None, None, None), slice(None, 10, None))];  w_1 = None
        tmp_12 = tmp_11.expand(1, 10);  tmp_11 = None
        tmp_13 = w_0[(slice(None, None, None), slice(0, 10, None))];  w_0 = None
        tmp_14 = torch.nn.functional.embedding(in_1, w_6, 0, None, 2.0, False, False);  in_1 = w_6 = None
        tmp_15 = torch.nn.functional.embedding(tmp_12, w_5, None, None, 2.0, False, False);  tmp_12 = w_5 = None
        tmp_16 = tmp_14 + tmp_15;  tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.embedding(tmp_13, w_4, None, None, 2.0, False, False);  tmp_13 = w_4 = None
        tmp_16 += tmp_17;  tmp_18 = tmp_16;  tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (768,), w_3, w_2, 1e-12);  tmp_18 = w_3 = w_2 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.1, False, False);  tmp_19 = None
        tmp_21 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_22 = tmp_21.expand(1, 1, 10, 10);  tmp_21 = None
        tmp_23 = tmp_22.to(torch.float32);  tmp_22 = None
        tmp_24 = torch.tensor(1.0, dtype = torch.float32)
        tmp_25 = tmp_24 - tmp_23;  tmp_24 = tmp_23 = None
        tmp_26 = tmp_25.to(torch.bool)
        tmp_27 = tmp_25.masked_fill(tmp_26, -3.4028234663852886e+38);  tmp_25 = tmp_26 = None
        linear = torch.nn.functional.linear(tmp_20, w_8, w_7);  w_8 = w_7 = None
        tmp_29 = linear.view(1, -1, 12, 64);  linear = None
        return (tmp_20, tmp_27, tmp_29)
        