import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.nn.functional.relu(in_1, inplace = False);  in_1 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, p = 0.0, training = False);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, w_3, w_2);  tmp_7 = w_3 = w_2 = None
        tmp_9 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_10 = in_0 + tmp_9;  in_0 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), w_5, w_4, 1e-05);  tmp_10 = w_5 = w_4 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (256,), w_1, w_0, 1e-05);  tmp_11 = w_1 = w_0 = None
        return (tmp_12,)
        