import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor):
        tmp_13 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_14 = tmp_13.to(dtype = torch.float32);  tmp_13 = None
        tmp_15 = 1.0 - tmp_14;  tmp_14 = None
        tmp_16 = tmp_15 * -3.4028234663852886e+38;  tmp_15 = None
        tmp_17 = in_2[(slice(None, None, None), slice(0, 64, None))];  in_2 = None
        tmp_18 = torch.nn.functional.embedding(in_1, in_7, 0, None, 2.0, False, False);  in_1 = in_7 = None
        tmp_19 = torch.nn.functional.embedding(in_12, in_6, None, None, 2.0, False, False);  in_12 = in_6 = None
        tmp_20 = tmp_18 + tmp_19;  tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.embedding(tmp_17, in_5, None, None, 2.0, False, False);  tmp_17 = in_5 = None
        tmp_20 += tmp_21;  tmp_22 = tmp_20;  tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (128,), in_4, in_3, 1e-12);  tmp_22 = in_4 = in_3 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False);  tmp_23 = None
        linear = torch.nn.functional.linear(tmp_24, in_9, in_8);  tmp_24 = in_9 = in_8 = None
        linear_1 = torch.nn.functional.linear(linear, in_11, in_10);  in_11 = in_10 = None
        return (tmp_16, linear, linear_1)
        