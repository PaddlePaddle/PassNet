import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_1 = torch.cat((in_0, tmp_0), dim=1)
        tmp_0 = None
        tmp_2 = tmp_1.view(1, 2, 488, 7, 7)
        tmp_1 = None
        tmp_3 = torch.transpose(tmp_2, 1, 2)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 976, 7, 7)
        tmp_4 = None
        tmp_6 = tmp_5.chunk(2, dim=1)
        tmp_5 = None
        tmp_7 = tmp_6[0]
        tmp_8 = tmp_6[1]
        tmp_6 = None
        return (tmp_7, tmp_8)