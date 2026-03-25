import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_1 = tmp_0.view(1, 512, 64, 64)
        tmp_0 = None
        tmp_2 = in_0.view(512, 1, 3, 3)
        return (tmp_2, tmp_1)