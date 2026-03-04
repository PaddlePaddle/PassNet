import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 / 11.313708498984761
        tmp_1 = torch.nn.functional.relu(tmp_0)
        tmp_0 = None
        tmp_2 = torch.square(tmp_1)
        tmp_1 = None
        return (tmp_2,)