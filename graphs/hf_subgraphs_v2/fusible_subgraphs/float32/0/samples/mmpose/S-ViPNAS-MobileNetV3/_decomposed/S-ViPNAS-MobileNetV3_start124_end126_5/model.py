import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.hardswish(in_0, True)
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
        return (tmp_1, tmp_0)