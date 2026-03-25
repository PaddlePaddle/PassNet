import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, 8)
        tmp_1 = tmp_0.view(2, 256, -1)
        tmp_0 = None
        return (tmp_1,)