import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor):
        tmp_10 = in_2[(slice(None, None, None), slice(None, 256, None))];  in_2 = None
        tmp_11 = tmp_10.expand(1, 256);  tmp_10 = None
        tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_13 = tmp_12.to(dtype = torch.float32);  tmp_12 = None
        tmp_14 = 1.0 - tmp_13;  tmp_13 = None
        tmp_15 = tmp_14 * -3.4028234663852886e+38;  tmp_14 = None
        tmp_16 = in_1.ne(1)
        tmp_17 = tmp_16.int();  tmp_16 = None
        tmp_18 = torch.cumsum(tmp_17, dim = 1)
        tmp_19 = tmp_18.type_as(tmp_17);  tmp_18 = None
        tmp_20 = tmp_19 + 0;  tmp_19 = None
        tmp_21 = tmp_20 * tmp_17;  tmp_20 = tmp_17 = None
        tmp_22 = tmp_21.long();  tmp_21 = None
        tmp_23 = tmp_22 + 1;  tmp_22 = None
        tmp_24 = torch.nn.functional.embedding(in_1, in_7, 1, None, 2.0, False, False);  in_1 = in_7 = None
        tmp_25 = torch.nn.functional.embedding(tmp_11, in_6, None, None, 2.0, False, False);  tmp_11 = in_6 = None
        tmp_26 = tmp_24 + tmp_25;  tmp_24 = tmp_25 = None
        tmp_27 = torch.nn.functional.embedding(tmp_23, in_5, 1, None, 2.0, False, False);  tmp_23 = in_5 = None
        tmp_26 += tmp_27;  tmp_28 = tmp_26;  tmp_26 = tmp_27 = None
        tmp_29 = torch.nn.functional.layer_norm(tmp_28, (32,), in_4, in_3, 1e-12);  tmp_28 = in_4 = in_3 = None
        tmp_30 = torch.nn.functional.dropout(tmp_29, 0.1, False, False);  tmp_29 = None
        linear = torch.nn.functional.linear(tmp_30, in_9, in_8);  in_9 = in_8 = None
        return (tmp_30, tmp_15, linear)
        