import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
        tmp_1 = tmp_0 + in_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
        tmp_1 = None
        return (tmp_2,)