import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = in_1 * 0.25
        tmp_1 = in_0.reshape(-1, 8, 8, 16)
        tmp_2 = w_0.transpose(-1, -2)
        return (tmp_0, tmp_1, tmp_2)