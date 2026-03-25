import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = in_2 + in_3;  in_2 = in_3 = None
        tmp_3 = tmp_2[(slice(None, None, None), 0)];  tmp_2 = None
        tmp_4 = in_0.view(-1, 7);  in_0 = None
        tmp_5 = in_1[(slice(None, None, None), slice(None, 7, None))];  in_1 = None
        return (tmp_4, tmp_3, tmp_5)
        