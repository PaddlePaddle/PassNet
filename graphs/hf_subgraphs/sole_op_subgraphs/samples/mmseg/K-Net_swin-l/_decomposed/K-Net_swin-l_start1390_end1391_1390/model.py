import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.pad(in_0, (0, 0, 0, 5, 0, 5), 'constant', None)
        return (tmp_0,)