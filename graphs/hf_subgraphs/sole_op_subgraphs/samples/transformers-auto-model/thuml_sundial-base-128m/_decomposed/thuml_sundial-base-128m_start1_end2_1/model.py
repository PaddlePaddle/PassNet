import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.pad(tmp_0, (0, 0), 'constant', None)
        tmp_0 = None
        return (tmp_1,)