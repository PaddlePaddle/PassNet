import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.cat((in_0, in_1), dim=1)
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
        return (tmp_1, tmp_0)