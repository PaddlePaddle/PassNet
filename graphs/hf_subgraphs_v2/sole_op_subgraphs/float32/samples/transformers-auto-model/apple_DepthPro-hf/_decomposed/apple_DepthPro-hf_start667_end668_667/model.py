import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.split_with_sizes(in_0, [25, 9, 1])
        return (tmp_0,)