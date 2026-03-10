import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.cat([in_1, in_2, in_0], 2)
        return (tmp_0,)