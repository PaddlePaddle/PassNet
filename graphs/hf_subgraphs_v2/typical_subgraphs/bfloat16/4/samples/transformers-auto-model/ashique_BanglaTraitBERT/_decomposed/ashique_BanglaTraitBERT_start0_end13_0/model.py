import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        tmp_11 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_12 = tmp_11.to(dtype = torch.float32);  tmp_11 = None
        tmp_13 = 1.0 - tmp_12;  tmp_12 = None
        tmp_14 = tmp_13 * -3.4028234663852886e+38;  tmp_13 = None
        tmp_15 = in_2[(slice(None, None, None), slice(0, 256, None))];  in_2 = None
        tmp_16 = torch.nn.functional.embedding(in_1, in_7, 0, None, 2.0, False, False);  in_1 = in_7 = None
        tmp_17 = torch.nn.functional.embedding(in_10, in_6, None, None, 2.0, False, False);  in_10 = in_6 = None
        tmp_18 = tmp_16 + tmp_17;  tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.embedding(tmp_15, in_5, None, None, 2.0, False, False);  tmp_15 = in_5 = None
        tmp_18 += tmp_19;  tmp_20 = tmp_18;  tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (1024,), in_4, in_3, 1e-12);  tmp_20 = in_4 = in_3 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.1, False, False);  tmp_21 = None
        linear = torch.nn.functional.linear(tmp_22, in_9, in_8);  in_9 = in_8 = None
        return (tmp_22, tmp_14, linear)
        