import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.cat((in_1, in_0), dim=1)
        tmp_1 = tmp_0.view(1, 2, 176, 16, 16)
        tmp_0 = None
        tmp_2 = torch.transpose(tmp_1, 1, 2)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = tmp_3.view(1, 352, 16, 16)
        tmp_3 = None
        tmp_5 = tmp_4.chunk(2, dim=1)
        tmp_4 = None
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_5 = None
        return (tmp_6, tmp_7)