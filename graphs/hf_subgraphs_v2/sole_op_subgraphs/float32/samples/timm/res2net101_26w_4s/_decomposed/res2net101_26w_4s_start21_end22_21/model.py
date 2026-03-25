import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.avg_pool2d(in_0, 3, 1, 1, False, True, None)
        return (tmp_0,)