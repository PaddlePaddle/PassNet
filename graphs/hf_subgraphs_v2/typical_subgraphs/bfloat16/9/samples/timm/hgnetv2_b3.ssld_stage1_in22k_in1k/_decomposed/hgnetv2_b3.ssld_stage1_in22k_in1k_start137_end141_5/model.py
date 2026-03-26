import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = False);  in_2 = None
        tmp_3 = w_1 * tmp_2;  w_1 = tmp_2 = None
        tmp_4 = tmp_3 + w_0;  tmp_3 = w_0 = None
        tmp_5 = torch.cat([in_3, in_4, in_5, in_0, in_1, tmp_4], dim = 1);  in_3 = in_4 = in_5 = in_0 = in_1 = tmp_4 = None
        return (tmp_5,)
        