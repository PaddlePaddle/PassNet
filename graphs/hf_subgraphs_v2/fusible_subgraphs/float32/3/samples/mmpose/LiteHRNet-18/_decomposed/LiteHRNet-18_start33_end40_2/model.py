import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
        tmp_1 = in_0.chunk(2, dim=1)
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        tmp_4 = tmp_0.chunk(2, dim=1)
        tmp_0 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_4 = None
        return (tmp_3, tmp_2, tmp_5, tmp_6)