import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.avg_pool2d(in_0, 1, 1, 0, True, False, None)
        return (tmp_0,)