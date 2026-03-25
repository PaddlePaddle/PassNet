import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0 : torch.Tensor):
        tmp_1 = torch.rms_norm(in_0, (2048,), w_0, 1e-05);  in_0 = w_0 = None
        return (tmp_1,)
        