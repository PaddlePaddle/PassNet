import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 197, 3, 4, 48)
        tmp_1 = None
        tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
        tmp_2 = None
        tmp_4 = tmp_3.unbind(0)
        tmp_3 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_7 = tmp_4[2]
        tmp_4 = None
        tmp_8 = tmp_6.transpose(-2, -1)
        tmp_6 = None
        return (tmp_5, tmp_8, tmp_7)