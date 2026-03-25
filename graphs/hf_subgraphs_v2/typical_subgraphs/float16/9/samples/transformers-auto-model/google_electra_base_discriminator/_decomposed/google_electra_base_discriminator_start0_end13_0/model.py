import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, in_2 : torch.Tensor):
        tmp_11 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_12 = tmp_11.to(dtype = torch.float32);  tmp_11 = None
        tmp_13 = 1.0 - tmp_12;  tmp_12 = None
        tmp_14 = tmp_13 * -3.4028234663852886e+38;  tmp_13 = None
        tmp_15 = w_0[(slice(None, None, None), slice(0, 16, None))];  w_0 = None
        tmp_16 = torch.nn.functional.embedding(in_1, w_5, 0, None, 2.0, False, False);  in_1 = w_5 = None
        tmp_17 = torch.nn.functional.embedding(in_2, w_4, None, None, 2.0, False, False);  in_2 = w_4 = None
        tmp_18 = tmp_16 + tmp_17;  tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.embedding(tmp_15, w_3, None, None, 2.0, False, False);  tmp_15 = w_3 = None
        tmp_18 += tmp_19;  tmp_20 = tmp_18;  tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (768,), w_2, w_1, 1e-12);  tmp_20 = w_2 = w_1 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.1, False, False);  tmp_21 = None
        linear = torch.nn.functional.linear(tmp_22, w_7, w_6);  w_7 = w_6 = None
        return (tmp_22, tmp_14, linear)
        