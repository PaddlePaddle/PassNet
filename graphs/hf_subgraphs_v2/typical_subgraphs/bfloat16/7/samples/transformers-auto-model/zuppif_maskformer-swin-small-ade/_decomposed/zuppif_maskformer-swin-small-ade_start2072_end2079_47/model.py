import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = torch.nn.functional.relu(in_7, inplace = False);  in_7 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, p = 0.0, training = False);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, in_3, in_2);  tmp_7 = in_3 = in_2 = None
        tmp_9 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_10 = in_6 + tmp_9;  in_6 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_5, in_4, 1e-05);  tmp_10 = in_5 = in_4 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (256,), in_1, in_0, 1e-05);  tmp_11 = in_1 = in_0 = None
        return (tmp_12,)
        