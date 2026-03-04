import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.pad(in_0, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
        tmp_1 = in_1[slice(None, None, None), slice(None, -1, None)]
        return (tmp_1, tmp_0)