import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_2 = torch.nn.functional.relu(in_5, inplace = True);  in_5 = None
        tmp_3 = torch.cat((in_4, in_2, in_3, tmp_2), 1);  in_4 = in_2 = in_3 = tmp_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1);  tmp_3 = None
        tmp_5 = tmp_4.flatten(1, -1);  tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False);  tmp_5 = None
        to = tmp_6.to(torch.float16);  tmp_6 = None
        linear = torch.nn.functional.linear(to, in_1, in_0);  to = in_1 = in_0 = None
        return (linear,)
        