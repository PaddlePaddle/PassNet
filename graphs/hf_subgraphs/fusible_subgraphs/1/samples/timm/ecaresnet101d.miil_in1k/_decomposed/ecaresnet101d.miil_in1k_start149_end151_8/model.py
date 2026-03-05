import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.sym_sum([1, in_0])
        tmp_0 = None
        tmp_1 = in_1.view(1, 1, -1)
        return (tmp_1,)