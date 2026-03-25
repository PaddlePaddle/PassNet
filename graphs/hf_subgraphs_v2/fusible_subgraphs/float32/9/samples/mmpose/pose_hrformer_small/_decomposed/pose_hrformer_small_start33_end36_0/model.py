import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_1 = in_0.view(1, 32, -1)
        tmp_2 = tmp_1.permute(0, 2, 1)
        tmp_1 = None
        return (tmp_0, tmp_2)