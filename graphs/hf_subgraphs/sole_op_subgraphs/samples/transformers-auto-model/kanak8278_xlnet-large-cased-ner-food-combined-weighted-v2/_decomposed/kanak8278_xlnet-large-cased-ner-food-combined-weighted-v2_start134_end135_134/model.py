import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.index_select(in_1, 3, in_0)
        return (tmp_0,)