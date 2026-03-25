import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.cat((in_0, in_1), dim = 1);  in_0 = in_1 = None
        tmp_1 = tmp_0.view(128, 2, 116, 14, 14);  tmp_0 = None
        tmp_2 = torch.transpose(tmp_1, 1, 2);  tmp_1 = None
        tmp_3 = tmp_2.contiguous();  tmp_2 = None
        tmp_4 = tmp_3.view(128, 232, 14, 14);  tmp_3 = None
        return (tmp_4,)
        