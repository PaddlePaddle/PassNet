import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor):
        tmp_10 = torch.nn.functional.dropout(in_8, p = 0.1, training = False);  in_8 = None
        tmp_11 = in_9 + tmp_10;  in_9 = tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (256,), in_1, in_0, 1e-05);  tmp_11 = in_1 = in_0 = None
        linear = torch.nn.functional.linear(tmp_12, in_3, in_2);  in_3 = in_2 = None
        tmp_14 = torch.nn.functional.relu(linear, inplace = False);  linear = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, p = 0.0, training = False);  tmp_14 = None
        linear_1 = torch.nn.functional.linear(tmp_15, in_5, in_4);  tmp_15 = in_5 = in_4 = None
        tmp_17 = torch.nn.functional.dropout(linear_1, p = 0.1, training = False);  linear_1 = None
        tmp_18 = tmp_12 + tmp_17;  tmp_12 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (256,), in_7, in_6, 1e-05);  tmp_18 = in_7 = in_6 = None
        return (tmp_19,)
        