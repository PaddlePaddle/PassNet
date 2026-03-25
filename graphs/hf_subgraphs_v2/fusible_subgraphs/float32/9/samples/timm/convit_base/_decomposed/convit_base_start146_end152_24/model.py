import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.linear(in_2, tmp_0, None)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 196, 2, 16, 48)
        tmp_1 = None
        tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
        tmp_2 = None
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_3 = None
        tmp_6 = in_1.expand(1, -1, -1, -1)
        return (tmp_5, tmp_6, tmp_4)