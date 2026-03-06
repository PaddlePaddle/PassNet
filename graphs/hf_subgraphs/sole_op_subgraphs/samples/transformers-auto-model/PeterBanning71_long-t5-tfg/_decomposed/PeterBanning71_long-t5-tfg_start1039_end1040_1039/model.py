import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.where(in_2, in_1, in_0)
        return (tmp_0,)