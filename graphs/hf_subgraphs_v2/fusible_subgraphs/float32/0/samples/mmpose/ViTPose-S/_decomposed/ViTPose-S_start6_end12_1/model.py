import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.reshape(1, 192, 3, 12, 32)
        tmp_2 = None
        tmp_4 = tmp_3.permute(2, 0, 3, 1, 4)
        tmp_3 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_7 = tmp_4[2]
        tmp_4 = None
        return (tmp_6, tmp_5, tmp_7)