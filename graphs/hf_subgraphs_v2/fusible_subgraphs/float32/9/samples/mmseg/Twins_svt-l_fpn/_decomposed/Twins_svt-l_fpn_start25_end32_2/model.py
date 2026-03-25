import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.reshape(1, 361, 49, 3, 4, 32)
        tmp_2 = None
        tmp_4 = tmp_3.permute(3, 0, 1, 4, 2, 5)
        tmp_3 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_7 = tmp_4[2]
        tmp_4 = None
        tmp_8 = tmp_6.transpose(-2, -1)
        tmp_6 = None
        return (tmp_5, tmp_8, tmp_7)