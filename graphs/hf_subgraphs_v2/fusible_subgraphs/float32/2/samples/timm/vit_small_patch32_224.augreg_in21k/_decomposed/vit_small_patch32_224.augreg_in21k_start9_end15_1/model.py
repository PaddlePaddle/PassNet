import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 50, 3, 6, 64)
        tmp_1 = tmp_0.permute(2, 0, 3, 1, 4)
        tmp_0 = None
        tmp_2 = tmp_1.unbind(0)
        tmp_1 = None
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_5 = tmp_2[2]
        tmp_2 = None
        return (tmp_4, tmp_3, tmp_5)