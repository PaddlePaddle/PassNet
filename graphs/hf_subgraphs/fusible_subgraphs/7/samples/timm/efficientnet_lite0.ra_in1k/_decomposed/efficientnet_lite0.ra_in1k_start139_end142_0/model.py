import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
        tmp_0 = None
        tmp_2 = tmp_1.flatten(1, -1)
        tmp_1 = None
        return (tmp_2,)