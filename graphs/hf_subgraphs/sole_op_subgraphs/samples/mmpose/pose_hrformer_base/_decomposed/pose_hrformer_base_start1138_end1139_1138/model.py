import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.pad(in_0, (0, 0, 2, 2, 1, 2), 'constant', None)
        return (tmp_0,)