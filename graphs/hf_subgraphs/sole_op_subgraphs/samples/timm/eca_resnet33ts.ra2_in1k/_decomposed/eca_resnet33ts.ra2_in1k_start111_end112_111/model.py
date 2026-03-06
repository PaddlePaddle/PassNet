import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.sym_sum([1, in_0])
        tmp_0 = None
        return ()