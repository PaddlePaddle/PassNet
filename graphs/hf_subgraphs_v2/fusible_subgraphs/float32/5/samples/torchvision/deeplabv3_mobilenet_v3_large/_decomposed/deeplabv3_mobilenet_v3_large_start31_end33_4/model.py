import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
        return (tmp_0, tmp_1)