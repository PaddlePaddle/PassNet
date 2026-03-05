import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 * 0.45
        tmp_1 = tmp_0 + -0.18799999999999994
        tmp_0 = None
        return (tmp_1,)