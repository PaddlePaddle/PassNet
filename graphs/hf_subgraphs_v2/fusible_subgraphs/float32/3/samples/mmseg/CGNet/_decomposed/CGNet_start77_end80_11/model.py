import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
        tmp_3 = tmp_2.view(32, 128)
        tmp_2 = None
        return (tmp_1, tmp_3)