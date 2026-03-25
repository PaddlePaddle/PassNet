import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_2 = torch.nn.functional.relu(in_7, inplace = False);  in_7 = None
        tmp_3 = in_1 * tmp_2;  in_1 = tmp_2 = None
        tmp_4 = tmp_3 + in_0;  tmp_3 = in_0 = None
        tmp_5 = torch.cat([in_2, in_3, in_4, in_5, in_6, tmp_4], dim = 1);  in_2 = in_3 = in_4 = in_5 = in_6 = tmp_4 = None
        return (tmp_5,)
        