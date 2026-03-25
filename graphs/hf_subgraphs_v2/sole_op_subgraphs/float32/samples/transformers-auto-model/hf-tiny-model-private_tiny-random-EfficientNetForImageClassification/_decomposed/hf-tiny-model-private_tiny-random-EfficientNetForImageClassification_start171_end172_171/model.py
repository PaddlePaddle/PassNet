import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.pad(in_0, (1, 2, 1, 2), 'constant', in_1)
        return (tmp_0,)