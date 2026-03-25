import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_0 = torch.sigmoid(in_0);  in_0 = None
        tmp_1 = tmp_0.view(1, 64, 1, 1);  tmp_0 = None
        tmp_2 = in_1 * tmp_1;  in_1 = tmp_1 = None
        tmp_3 = in_2 + tmp_2;  in_2 = tmp_2 = None
        return (tmp_3,)
        