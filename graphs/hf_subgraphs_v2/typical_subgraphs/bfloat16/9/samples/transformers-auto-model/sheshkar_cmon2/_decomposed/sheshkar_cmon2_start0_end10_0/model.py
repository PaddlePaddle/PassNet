import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_10 = torch.nn.functional.dropout(in_0, p = 0.1, training = False);  in_0 = None
        tmp_11 = in_1 + tmp_10;  in_1 = tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (256,), w_1, w_0, 1e-05);  tmp_11 = w_1 = w_0 = None
        linear = torch.nn.functional.linear(tmp_12, w_3, w_2);  w_3 = w_2 = None
        tmp_14 = torch.nn.functional.relu(linear, inplace = False);  linear = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, p = 0.0, training = False);  tmp_14 = None
        linear_1 = torch.nn.functional.linear(tmp_15, w_5, w_4);  tmp_15 = w_5 = w_4 = None
        tmp_17 = torch.nn.functional.dropout(linear_1, p = 0.1, training = False);  linear_1 = None
        tmp_18 = tmp_12 + tmp_17;  tmp_12 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (256,), w_7, w_6, 1e-05);  tmp_18 = w_7 = w_6 = None
        return (tmp_19,)
        