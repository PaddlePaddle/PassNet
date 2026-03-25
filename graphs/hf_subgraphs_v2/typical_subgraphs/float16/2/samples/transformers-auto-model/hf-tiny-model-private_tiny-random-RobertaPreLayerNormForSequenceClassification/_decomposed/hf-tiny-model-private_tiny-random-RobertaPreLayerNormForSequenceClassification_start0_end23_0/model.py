import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor):
        tmp_12 = in_2[(slice(None, None, None), slice(None, 128, None))];  in_2 = None
        tmp_13 = tmp_12.expand(1, 128);  tmp_12 = None
        tmp_14 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_15 = tmp_14.to(dtype = torch.float32);  tmp_14 = None
        tmp_16 = 1.0 - tmp_15;  tmp_15 = None
        tmp_17 = tmp_16 * -3.4028234663852886e+38;  tmp_16 = None
        tmp_18 = in_1.ne(1)
        tmp_19 = tmp_18.int();  tmp_18 = None
        tmp_20 = torch.cumsum(tmp_19, dim = 1)
        tmp_21 = tmp_20.type_as(tmp_19);  tmp_20 = None
        tmp_22 = tmp_21 + 0;  tmp_21 = None
        tmp_23 = tmp_22 * tmp_19;  tmp_22 = tmp_19 = None
        tmp_24 = tmp_23.long();  tmp_23 = None
        tmp_25 = tmp_24 + 1;  tmp_24 = None
        tmp_26 = torch.nn.functional.embedding(in_1, in_7, 1, None, 2.0, False, False);  in_1 = in_7 = None
        tmp_27 = torch.nn.functional.embedding(tmp_13, in_6, None, None, 2.0, False, False);  tmp_13 = in_6 = None
        tmp_28 = tmp_26 + tmp_27;  tmp_26 = tmp_27 = None
        tmp_29 = torch.nn.functional.embedding(tmp_25, in_5, 1, None, 2.0, False, False);  tmp_25 = in_5 = None
        tmp_28 += tmp_29;  tmp_30 = tmp_28;  tmp_28 = tmp_29 = None
        tmp_31 = torch.nn.functional.layer_norm(tmp_30, (32,), in_4, in_3, 1e-12);  tmp_30 = in_4 = in_3 = None
        tmp_32 = torch.nn.functional.dropout(tmp_31, 0.1, False, False);  tmp_31 = None
        tmp_33 = torch.nn.functional.layer_norm(tmp_32, (32,), in_9, in_8, 1e-12);  in_9 = in_8 = None
        linear = torch.nn.functional.linear(tmp_33, in_11, in_10);  in_11 = in_10 = None
        return (tmp_32, tmp_17, tmp_33, linear)
        