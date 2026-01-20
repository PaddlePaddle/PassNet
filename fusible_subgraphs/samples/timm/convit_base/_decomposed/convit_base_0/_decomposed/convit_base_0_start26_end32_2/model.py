import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = torch.nn.functional.linear(in_0, w_0, None)
        tmp_1 = tmp_0.reshape(1, 196, 2, 16, 48)
        tmp_0 = None
        tmp_2 = tmp_1.permute(2, 0, 3, 1, 4)
        tmp_1 = None
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_2 = None
        tmp_5 = in_1.expand(1, -1, -1, -1)
        return (tmp_3, tmp_4, tmp_5)