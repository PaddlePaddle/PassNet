import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor):
        tmp_19 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_20 = tmp_19.to(dtype = torch.float32);  tmp_19 = None
        tmp_21 = 1.0 - tmp_20;  tmp_20 = None
        tmp_22 = tmp_21 * -3.4028234663852886e+38;  tmp_21 = None
        tmp_23 = in_2[(slice(None, None, None), slice(None, 128, None))];  in_2 = None
        tmp_24 = torch.nn.functional.embedding(in_1, in_9, 0, None, 2.0, False, False);  in_1 = in_9 = None
        tmp_25 = tmp_24[(slice(None, None, None), slice(1, None, None))]
        tmp_26 = torch.nn.functional.pad(tmp_25, [0, 0, 0, 1, 0, 0], 'constant', 0.0);  tmp_25 = None
        tmp_27 = tmp_24[(slice(None, None, None), slice(None, -1, None))]
        tmp_28 = torch.nn.functional.pad(tmp_27, [0, 0, 1, 0, 0, 0], 'constant', 0.0);  tmp_27 = None
        tmp_29 = torch.cat([tmp_26, tmp_24, tmp_28], dim = 2);  tmp_26 = tmp_24 = tmp_28 = None
        linear = torch.nn.functional.linear(tmp_29, in_6, in_5);  tmp_29 = in_6 = in_5 = None
        tmp_31 = torch.nn.functional.embedding(tmp_23, in_7, None, None, 2.0, False, False);  tmp_23 = in_7 = None
        tmp_32 = torch.nn.functional.embedding(in_18, in_8, None, None, 2.0, False, False);  in_18 = in_8 = None
        tmp_33 = linear + tmp_31;  linear = tmp_31 = None
        tmp_34 = tmp_33 + tmp_32;  tmp_33 = tmp_32 = None
        tmp_35 = tmp_34 * in_4;  tmp_34 = in_4 = None
        tmp_36 = tmp_35 + in_3;  tmp_35 = in_3 = None
        tmp_37 = torch.nn.functional.dropout(tmp_36, 0.0, False, False);  tmp_36 = None
        linear_1 = torch.nn.functional.linear(tmp_37, in_17, in_16);  in_17 = in_16 = None
        tmp_39 = linear_1 * in_15;  linear_1 = in_15 = None
        tmp_40 = tmp_39 + in_14;  tmp_39 = in_14 = None
        linear_2 = torch.nn.functional.linear(tmp_37, in_13, in_12);  in_13 = in_12 = None
        tmp_42 = linear_2 * in_11;  linear_2 = in_11 = None
        tmp_43 = tmp_42 + in_10;  tmp_42 = in_10 = None
        return (tmp_37, tmp_22, tmp_40, tmp_43)
        