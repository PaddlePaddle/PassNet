import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, in_2 : torch.Tensor):
        tmp_19 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_20 = tmp_19.to(dtype = torch.float32);  tmp_19 = None
        tmp_21 = 1.0 - tmp_20;  tmp_20 = None
        tmp_22 = tmp_21 * -3.4028234663852886e+38;  tmp_21 = None
        tmp_23 = w_0[(slice(None, None, None), slice(None, 20, None))];  w_0 = None
        tmp_24 = torch.nn.functional.embedding(in_1, w_7, 0, None, 2.0, False, False);  in_1 = w_7 = None
        tmp_25 = tmp_24[(slice(None, None, None), slice(1, None, None))]
        tmp_26 = torch.nn.functional.pad(tmp_25, [0, 0, 0, 1, 0, 0], 'constant', 0.0);  tmp_25 = None
        tmp_27 = tmp_24[(slice(None, None, None), slice(None, -1, None))]
        tmp_28 = torch.nn.functional.pad(tmp_27, [0, 0, 1, 0, 0, 0], 'constant', 0.0);  tmp_27 = None
        tmp_29 = torch.cat([tmp_26, tmp_24, tmp_28], dim = 2);  tmp_26 = tmp_24 = tmp_28 = None
        linear = torch.nn.functional.linear(tmp_29, w_4, w_3);  tmp_29 = w_4 = w_3 = None
        tmp_31 = torch.nn.functional.embedding(tmp_23, w_5, None, None, 2.0, False, False);  tmp_23 = w_5 = None
        tmp_32 = torch.nn.functional.embedding(in_2, w_6, None, None, 2.0, False, False);  in_2 = w_6 = None
        tmp_33 = linear + tmp_31;  linear = tmp_31 = None
        tmp_34 = tmp_33 + tmp_32;  tmp_33 = tmp_32 = None
        tmp_35 = tmp_34 * w_2;  tmp_34 = w_2 = None
        tmp_36 = tmp_35 + w_1;  tmp_35 = w_1 = None
        tmp_37 = torch.nn.functional.dropout(tmp_36, 0.0, False, False);  tmp_36 = None
        linear_1 = torch.nn.functional.linear(tmp_37, w_15, w_14);  w_15 = w_14 = None
        tmp_39 = linear_1 * w_13;  linear_1 = w_13 = None
        tmp_40 = tmp_39 + w_12;  tmp_39 = w_12 = None
        linear_2 = torch.nn.functional.linear(tmp_37, w_11, w_10);  w_11 = w_10 = None
        tmp_42 = linear_2 * w_9;  linear_2 = w_9 = None
        tmp_43 = tmp_42 + w_8;  tmp_42 = w_8 = None
        return (tmp_37, tmp_22, tmp_40, tmp_43)
        