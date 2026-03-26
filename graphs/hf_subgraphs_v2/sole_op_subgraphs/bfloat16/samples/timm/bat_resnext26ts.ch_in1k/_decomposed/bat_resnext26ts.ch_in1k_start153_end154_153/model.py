import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        tmp_0 = torch.cat((in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7), dim = 3);  in_0 = in_1 = in_2 = in_3 = in_4 = in_5 = in_6 = in_7 = None
        return (tmp_0,)
        