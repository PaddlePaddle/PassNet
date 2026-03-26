import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.contiguous();  in_0 = None
        tmp_1 = tmp_0.view(32, 512, 64, 128);  tmp_0 = None
        return (tmp_1,)
        