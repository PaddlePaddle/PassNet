import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = tmp_0.mean((2, 3))
        tmp_2 = tmp_1.view(1, 1, -1)
        tmp_1 = None
        return (tmp_0, tmp_2)