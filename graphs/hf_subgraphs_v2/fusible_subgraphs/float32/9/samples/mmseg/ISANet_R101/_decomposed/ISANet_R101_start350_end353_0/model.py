import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = tmp_0.view(1, 512, 8, 8, 8, 8)
        tmp_2 = tmp_1.permute(0, 3, 5, 1, 2, 4)
        tmp_1 = None
        return (tmp_0, tmp_2)