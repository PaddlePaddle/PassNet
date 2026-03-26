import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_3 = torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False);  in_1 = in_2 = None
        tmp_4 = in_0.long();  in_0 = None
        return (tmp_3, tmp_4)
        