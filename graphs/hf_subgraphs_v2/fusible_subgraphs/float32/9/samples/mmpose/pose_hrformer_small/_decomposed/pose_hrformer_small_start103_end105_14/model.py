import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 * 0.1767766952966369
        tmp_1 = in_0.transpose(-2, -1)
        return (tmp_0, tmp_1)