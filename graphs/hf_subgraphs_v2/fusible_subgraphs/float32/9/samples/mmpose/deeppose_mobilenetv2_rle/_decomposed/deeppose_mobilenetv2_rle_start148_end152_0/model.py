import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
        tmp_0 = None
        tmp_2 = tmp_1.view(1, -1)
        tmp_1 = None
        tmp_3 = torch.flatten(tmp_2, 1)
        tmp_2 = None
        return (tmp_3,)