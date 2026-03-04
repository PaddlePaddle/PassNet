import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(32, 192, 3, 12, 32)
        tmp_1 = tmp_0.permute(2, 0, 3, 1, 4)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_4 = tmp_1[2]
        tmp_1 = None
        return (tmp_3, tmp_2, tmp_4)