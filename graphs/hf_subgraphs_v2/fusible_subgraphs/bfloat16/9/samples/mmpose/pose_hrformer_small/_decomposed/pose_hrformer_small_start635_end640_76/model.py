import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        in_3 += in_0;  in_4 = in_3;  in_3 = in_0 = None
        in_4 += in_2;  tmp_0 = in_4;  in_4 = in_2 = None
        tmp_2 = torch.nn.functional.relu(tmp_0, inplace = True);  tmp_0 = None
        tmp_3 = in_1.view(1, 32, -1);  in_1 = None
        tmp_4 = tmp_3.permute(0, 2, 1);  tmp_3 = None
        return (tmp_2, tmp_4)
        