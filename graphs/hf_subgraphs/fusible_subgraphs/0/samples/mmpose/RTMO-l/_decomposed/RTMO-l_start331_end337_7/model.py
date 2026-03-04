import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1.reshape(1, 64, -1)
        tmp_1 = in_0 + tmp_0
        tmp_2 = in_0 + tmp_0
        tmp_0 = None
        tmp_3 = tmp_1.transpose(0, 1)
        tmp_1 = None
        tmp_4 = tmp_2.transpose(0, 1)
        tmp_2 = None
        tmp_5 = in_0.transpose(0, 1)
        return (tmp_4, tmp_3, tmp_5)