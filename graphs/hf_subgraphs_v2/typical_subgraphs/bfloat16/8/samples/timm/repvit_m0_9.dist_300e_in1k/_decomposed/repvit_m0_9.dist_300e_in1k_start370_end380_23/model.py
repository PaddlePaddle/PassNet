import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
        tmp_12 = in_13 + in_12;  in_13 = in_12 = None
        tmp_13 = tmp_12.mean((2, 3), keepdim = False);  tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  in_6 = in_7 = in_9 = in_8 = None
        linear = torch.nn.functional.linear(tmp_16, in_11, in_10);  tmp_16 = in_11 = in_10 = None
        tmp_18 = torch.nn.functional.batch_norm(tmp_15, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_15 = in_0 = in_1 = in_3 = in_2 = None
        linear_1 = torch.nn.functional.linear(tmp_18, in_5, in_4);  tmp_18 = in_5 = in_4 = None
        tmp_20 = linear + linear_1;  linear = linear_1 = None
        tmp_21 = tmp_20 / 2;  tmp_20 = None
        return (tmp_21,)
        