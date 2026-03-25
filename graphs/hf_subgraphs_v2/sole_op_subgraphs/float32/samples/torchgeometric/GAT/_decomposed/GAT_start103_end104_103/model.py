import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1.index_select(0, in_0)
        return (tmp_0,)