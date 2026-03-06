import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.flatten(in_0, 1)
        tmp_0 = None
        return ()