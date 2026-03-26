import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_1 = torch.sigmoid(tmp_0);  tmp_0 = None
        tmp_2 = in_0 * tmp_1;  tmp_1 = None
        tmp_3 = tmp_2 + in_0;  tmp_2 = in_0 = None
        return (tmp_3,)
        