import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
        tmp_1 = torch.cat([tmp_0, in_1], dim=1)
        tmp_0 = None
        return (tmp_1,)