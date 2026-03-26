import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1):
        tmp_12 = in_1 + in_0;  in_1 = in_0 = None
        tmp_13 = tmp_12.mean((2, 3), keepdim = False);  tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, w_6, w_7, w_9, w_8, False, 0.1, 1e-05);  w_6 = w_7 = w_9 = w_8 = None
        linear = torch.nn.functional.linear(tmp_16, w_11, w_10);  tmp_16 = w_11 = w_10 = None
        tmp_18 = torch.nn.functional.batch_norm(tmp_15, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_15 = w_0 = w_1 = w_3 = w_2 = None
        linear_1 = torch.nn.functional.linear(tmp_18, w_5, w_4);  tmp_18 = w_5 = w_4 = None
        tmp_20 = linear + linear_1;  linear = linear_1 = None
        tmp_21 = tmp_20 / 2;  tmp_20 = None
        return (tmp_21,)
        