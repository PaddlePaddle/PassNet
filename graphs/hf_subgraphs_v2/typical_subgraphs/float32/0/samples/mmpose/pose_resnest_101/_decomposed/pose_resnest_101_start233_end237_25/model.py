import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = tmp_0.view(1, 2, -1, 16, 12)
        tmp_0 = None
        tmp_2 = tmp_1.sum(dim=1)
        tmp_3 = torch.nn.functional.adaptive_avg_pool2d(tmp_2, 1)
        tmp_2 = None
        return (tmp_3, tmp_1)