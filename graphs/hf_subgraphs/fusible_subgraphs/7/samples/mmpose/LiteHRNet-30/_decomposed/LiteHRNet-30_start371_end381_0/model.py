import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
        tmp_1 = in_1.chunk(2, dim=1)
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        tmp_4 = in_2.chunk(2, dim=1)
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_4 = None
        tmp_7 = tmp_0.chunk(2, dim=1)
        tmp_0 = None
        tmp_8 = tmp_7[0]
        tmp_9 = tmp_7[1]
        tmp_7 = None
        return (tmp_2, tmp_5, tmp_8, tmp_3, tmp_6, tmp_9)