import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.linear(in_2, weight=tmp_0, bias=in_1)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 257, 3, 16, -1)
        tmp_1 = None
        tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
        tmp_2 = None
        tmp_4 = tmp_3.unbind(0)
        tmp_3 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_7 = tmp_4[2]
        tmp_4 = None
        return (tmp_6, tmp_5, tmp_7)