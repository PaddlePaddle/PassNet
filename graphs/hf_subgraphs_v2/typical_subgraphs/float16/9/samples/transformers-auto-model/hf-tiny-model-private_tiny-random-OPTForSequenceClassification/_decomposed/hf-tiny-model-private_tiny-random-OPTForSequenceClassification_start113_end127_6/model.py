import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_7, w_6);  in_0 = w_7 = w_6 = None
        tmp_13 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_14 = in_1 + tmp_13;  in_1 = tmp_13 = None
        tmp_15 = tmp_14.reshape(-1, 16);  tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (16,), w_5, w_4, 1e-05);  w_5 = w_4 = None
        linear_1 = torch.nn.functional.linear(tmp_16, w_1, w_0);  tmp_16 = w_1 = w_0 = None
        tmp_18 = torch.nn.functional.relu(linear_1, inplace = False);  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_18, w_3, w_2);  tmp_18 = w_3 = w_2 = None
        tmp_20 = torch.nn.functional.dropout(linear_2, p = 0.1, training = False);  linear_2 = None
        tmp_21 = tmp_15 + tmp_20;  tmp_15 = tmp_20 = None
        tmp_22 = tmp_21.view((1, 21, 16));  tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (16,), w_9, w_8, 1e-05);  w_9 = w_8 = None
        linear_3 = torch.nn.functional.linear(tmp_23, w_11, w_10);  w_11 = w_10 = None
        tmp_25 = linear_3 * 0.5;  linear_3 = None
        return (tmp_22, tmp_23, tmp_25)
        