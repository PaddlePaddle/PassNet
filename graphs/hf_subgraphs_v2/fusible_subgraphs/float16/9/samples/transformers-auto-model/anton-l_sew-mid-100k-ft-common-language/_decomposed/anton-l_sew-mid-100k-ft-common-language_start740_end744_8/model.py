import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.gelu(in_0);  in_0 = None
        tmp_1 = tmp_0.reshape(1, 124, 2, 768);  tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 248, 768);  tmp_1 = None
        tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None);  tmp_2 = None
        return (tmp_3,)
        