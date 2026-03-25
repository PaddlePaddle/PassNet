import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.cat((in_0, in_1), dim=1)
        tmp_1 = tmp_0.view(128, 2, 96, 8, 8)
        tmp_0 = None
        tmp_2 = torch.transpose(tmp_1, 1, 2)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = tmp_3.view(128, 192, 8, 8)
        tmp_3 = None
        tmp_5 = tmp_4.chunk(2, dim=1)
        tmp_4 = None
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_5 = None
        return (tmp_6, tmp_7)