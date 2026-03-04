import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.linear(in_3, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_4 = tmp_3.view(1, -1, 16, 64)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = tmp_0[slice(None, 3969, None)]
        tmp_0 = None
        tmp_7 = tmp_6.reshape(1, 63, 63, -1)
        tmp_6 = None
        tmp_8 = tmp_7.permute(0, 3, 1, 2)
        tmp_7 = None
        return (tmp_8, tmp_5)