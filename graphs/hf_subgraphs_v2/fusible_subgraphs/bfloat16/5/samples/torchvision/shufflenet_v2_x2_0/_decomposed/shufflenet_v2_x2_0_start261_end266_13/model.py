import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.cat((in_1, in_0), dim = 1);  in_1 = in_0 = None
        tmp_1 = tmp_0.view(32, 2, 488, 12, 12);  tmp_0 = None
        tmp_2 = torch.transpose(tmp_1, 1, 2);  tmp_1 = None
        tmp_3 = tmp_2.contiguous();  tmp_2 = None
        tmp_4 = tmp_3.view(32, 976, 12, 12);  tmp_3 = None
        return (tmp_4,)
        