import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.cat((in_1, in_0), dim = 1);  in_1 = in_0 = None
        tmp_1 = tmp_0.view(128, 2, 488, 8, 8);  tmp_0 = None
        tmp_2 = torch.transpose(tmp_1, 1, 2);  tmp_1 = None
        tmp_3 = tmp_2.contiguous();  tmp_2 = None
        tmp_4 = tmp_3.view(128, 976, 8, 8);  tmp_3 = None
        return (tmp_4,)
        