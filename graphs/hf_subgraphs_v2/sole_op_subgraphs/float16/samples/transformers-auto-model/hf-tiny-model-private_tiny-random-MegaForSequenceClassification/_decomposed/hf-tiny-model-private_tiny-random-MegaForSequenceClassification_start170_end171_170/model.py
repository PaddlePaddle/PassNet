import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.fft.irfft(in_0, n = 44);  in_0 = None
        return (tmp_0,)
        