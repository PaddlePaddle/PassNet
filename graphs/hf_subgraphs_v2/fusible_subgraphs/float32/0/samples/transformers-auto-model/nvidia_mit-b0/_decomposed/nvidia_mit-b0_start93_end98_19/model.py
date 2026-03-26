import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view(1, -1, 2, 32)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_2.permute(0, 2, 1)
        tmp_6 = tmp_5.reshape(1, 64, 64, 64)
        tmp_5 = None
        return (tmp_4, tmp_6)