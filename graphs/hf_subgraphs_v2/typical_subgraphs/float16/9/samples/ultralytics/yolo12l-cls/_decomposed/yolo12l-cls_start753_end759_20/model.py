import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = torch.nn.functional.silu(in_0, inplace = False);  in_0 = None
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, 1);  tmp_2 = None
        tmp_4 = tmp_3.flatten(1);  tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, True);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, w_1, w_0);  tmp_5 = w_1 = w_0 = None
        tmp_7 = linear.softmax(1)
        return (linear, tmp_7)
        