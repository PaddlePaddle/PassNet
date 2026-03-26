import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.pad(in_0, (1, 2), 'constant', 0)
        return (tmp_0,)