import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.fft.irfft(in_0, n=44)
        return (tmp_0,)