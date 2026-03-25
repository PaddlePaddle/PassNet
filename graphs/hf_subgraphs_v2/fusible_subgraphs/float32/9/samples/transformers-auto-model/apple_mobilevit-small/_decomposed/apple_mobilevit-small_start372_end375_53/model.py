import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(960, 4, 2, 2)
        tmp_1 = tmp_0.transpose(1, 2)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 240, 8, 8)
        tmp_1 = None
        return (tmp_2,)