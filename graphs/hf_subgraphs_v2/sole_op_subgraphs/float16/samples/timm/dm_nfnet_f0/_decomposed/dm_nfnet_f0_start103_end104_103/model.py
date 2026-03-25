import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.batch_norm(in_0, None, None, weight = in_1, training = True, momentum = 0.0, eps = 1e-05);  in_0 = in_1 = None
        return (tmp_0,)
        