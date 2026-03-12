import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.fft.fftn(in_0, dim=(1, 2))
        return (tmp_0,)