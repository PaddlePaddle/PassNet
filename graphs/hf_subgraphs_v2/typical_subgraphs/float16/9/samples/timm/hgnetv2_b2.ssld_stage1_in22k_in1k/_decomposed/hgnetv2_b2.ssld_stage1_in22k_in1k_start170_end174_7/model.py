import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1):
        tmp_2 = torch.nn.functional.relu(in_1, inplace = False);  in_1 = None
        tmp_3 = w_1 * tmp_2;  w_1 = tmp_2 = None
        tmp_4 = tmp_3 + w_0;  tmp_3 = w_0 = None
        tmp_5 = tmp_4 + in_0;  tmp_4 = in_0 = None
        return (tmp_5,)
        