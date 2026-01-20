import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = w_0.repeat(1, 1, 1)
        tmp_1 = tmp_0.transpose(1, 0)
        tmp_0 = None
        tmp_2 = in_0.transpose(1, 0)
        return (tmp_1, tmp_2)