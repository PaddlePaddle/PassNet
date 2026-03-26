import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1 * 0.25
        tmp_2 = in_2.reshape(4, 8, 8, -1)
        tmp_3 = tmp_0.transpose(-1, -2)
        tmp_0 = None
        return (tmp_1, tmp_2, tmp_3)