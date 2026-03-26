import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor):
        tmp_10 = in_1.ne(1)
        tmp_11 = tmp_10.int();  tmp_10 = None
        tmp_12 = torch.cumsum(tmp_11, dim = 1)
        tmp_13 = tmp_12.type_as(tmp_11);  tmp_12 = None
        tmp_14 = tmp_13 + 0;  tmp_13 = None
        tmp_15 = tmp_14 * tmp_11;  tmp_14 = tmp_11 = None
        tmp_16 = tmp_15.long();  tmp_15 = None
        tmp_17 = tmp_16 + 1;  tmp_16 = None
        tmp_18 = torch.nn.functional.embedding(in_1, in_6, 1, None, 2.0, False, False);  in_1 = in_6 = None
        tmp_19 = torch.nn.functional.embedding(in_9, in_5, None, None, 2.0, False, False);  in_9 = in_5 = None
        tmp_20 = tmp_18 + tmp_19;  tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.embedding(tmp_17, in_4, 1, None, 2.0, False, False);  tmp_17 = in_4 = None
        tmp_20 += tmp_21;  tmp_22 = tmp_20;  tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (1024,), in_3, in_2, 1e-05);  tmp_22 = in_3 = in_2 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False);  tmp_23 = None
        tmp_25 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_26 = tmp_25.expand(16, 1, 128, 128);  tmp_25 = None
        tmp_27 = tmp_26.to(torch.float32);  tmp_26 = None
        tmp_28 = torch.tensor(1.0, dtype = torch.float32)
        tmp_29 = tmp_28 - tmp_27;  tmp_28 = tmp_27 = None
        tmp_30 = tmp_29.to(torch.bool)
        tmp_31 = tmp_29.masked_fill(tmp_30, -3.4028234663852886e+38);  tmp_29 = tmp_30 = None
        linear = torch.nn.functional.linear(tmp_24, in_8, in_7);  in_8 = in_7 = None
        tmp_33 = linear.view(16, -1, 16, 64);  linear = None
        return (tmp_24, tmp_31, tmp_33)
        