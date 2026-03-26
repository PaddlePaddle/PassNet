import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_9, w_8);  in_0 = w_9 = w_8 = None
        tmp_11 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_12 = in_1 + tmp_11;  in_1 = tmp_11 = None
        tmp_13 = tmp_12.reshape(-1, 16);  tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (16,), w_7, w_6, 1e-05);  w_7 = w_6 = None
        linear_1 = torch.nn.functional.linear(tmp_14, w_3, w_2);  tmp_14 = w_3 = w_2 = None
        tmp_16 = torch.nn.functional.relu(linear_1, inplace = False);  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_16, w_5, w_4);  tmp_16 = w_5 = w_4 = None
        tmp_18 = torch.nn.functional.dropout(linear_2, p = 0.1, training = False);  linear_2 = None
        tmp_19 = tmp_13 + tmp_18;  tmp_13 = tmp_18 = None
        tmp_20 = tmp_19.view((1, 21, 16));  tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (16,), w_1, w_0, 1e-05);  tmp_20 = w_1 = w_0 = None
        return (tmp_21,)
        