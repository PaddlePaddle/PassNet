import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = tmp_0.view(1, 2, -1, 96, 72)
        tmp_0 = None
        return (tmp_1,)