import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1 * in_0;  in_1 = in_0 = None
        tmp_1 = torch.sum(tmp_0, dim = 1);  tmp_0 = None
        tmp_2 = tmp_1.unsqueeze(1);  tmp_1 = None
        tmp_3 = torch.sigmoid(tmp_2);  tmp_2 = None
        return (tmp_3,)
        