import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = torch.nn.functional.leaky_relu(in_3, 0.01, True);  in_3 = None
        tmp_3 = tmp_2 + in_2;  tmp_2 = in_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1);  tmp_3 = None
        tmp_5 = tmp_4.flatten(1, -1);  tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False);  tmp_5 = None
        to = tmp_6.to(torch.float16);  tmp_6 = None
        linear = torch.nn.functional.linear(to, in_1, in_0);  to = in_1 = in_0 = None
        return (linear,)
        