import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
        return (tmp_0,)