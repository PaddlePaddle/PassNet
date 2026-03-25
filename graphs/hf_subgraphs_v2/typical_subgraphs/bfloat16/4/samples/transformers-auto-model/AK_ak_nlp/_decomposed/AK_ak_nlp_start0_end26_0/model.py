import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor):
        tmp_10 = in_2[(slice(None, None, None), slice(None, 256, None))];  in_2 = None
        tmp_11 = tmp_10.expand(1, 256);  tmp_10 = None
        tmp_12 = in_1.ne(1)
        tmp_13 = tmp_12.int();  tmp_12 = None
        tmp_14 = torch.cumsum(tmp_13, dim = 1)
        tmp_15 = tmp_14.type_as(tmp_13);  tmp_14 = None
        tmp_16 = tmp_15 + 0;  tmp_15 = None
        tmp_17 = tmp_16 * tmp_13;  tmp_16 = tmp_13 = None
        tmp_18 = tmp_17.long();  tmp_17 = None
        tmp_19 = tmp_18 + 1;  tmp_18 = None
        tmp_20 = torch.nn.functional.embedding(in_1, in_7, 1, None, 2.0, False, False);  in_1 = in_7 = None
        tmp_21 = torch.nn.functional.embedding(tmp_11, in_6, None, None, 2.0, False, False);  tmp_11 = in_6 = None
        tmp_22 = tmp_20 + tmp_21;  tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.embedding(tmp_19, in_5, 1, None, 2.0, False, False);  tmp_19 = in_5 = None
        tmp_22 += tmp_23;  tmp_24 = tmp_22;  tmp_22 = tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (768,), in_4, in_3, 1e-12);  tmp_24 = in_4 = in_3 = None
        tmp_26 = torch.nn.functional.dropout(tmp_25, 0.1, False, False);  tmp_25 = None
        tmp_27 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_28 = tmp_27.expand(8, 1, 256, 256);  tmp_27 = None
        tmp_29 = tmp_28.to(torch.float32);  tmp_28 = None
        tmp_30 = torch.tensor(1.0, dtype = torch.float32)
        tmp_31 = tmp_30 - tmp_29;  tmp_30 = tmp_29 = None
        tmp_32 = tmp_31.to(torch.bool)
        tmp_33 = tmp_31.masked_fill(tmp_32, -3.4028234663852886e+38);  tmp_31 = tmp_32 = None
        linear = torch.nn.functional.linear(tmp_26, in_9, in_8);  in_9 = in_8 = None
        tmp_35 = linear.view(8, -1, 12, 64);  linear = None
        return (tmp_26, tmp_33, tmp_35)
        