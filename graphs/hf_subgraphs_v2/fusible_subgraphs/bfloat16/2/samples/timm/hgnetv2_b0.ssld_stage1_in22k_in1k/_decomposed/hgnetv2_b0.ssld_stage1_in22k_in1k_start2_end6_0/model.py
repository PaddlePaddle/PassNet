import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = False);  in_2 = None
        tmp_3 = in_1 * tmp_2;  in_1 = tmp_2 = None
        tmp_4 = tmp_3 + in_0;  tmp_3 = in_0 = None
        tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None);  tmp_4 = None
        return (tmp_5,)
        