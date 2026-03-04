import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 * 0.3535533905932738
        tmp_1 = in_0.transpose(-1, -2)
        return (tmp_0, tmp_1)