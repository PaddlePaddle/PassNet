import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, 1)
        tmp_1 = tmp_0.view(1, 64)
        tmp_0 = None
        return (tmp_1,)