import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        max_1 = torch.max(in_0, -1, keepdim = True)
        tmp_1 = max_1[0];  max_1 = None
        tmp_2 = tmp_1.expand_as(in_0);  tmp_1 = None
        tmp_3 = tmp_2 - in_0;  tmp_2 = in_0 = None
        tmp_4 = torch.nn.functional.softmax(tmp_3, dim = -1);  tmp_3 = None
        tmp_5 = in_1.view(2, 512, -1);  in_1 = None
        return (tmp_4, tmp_5)
        