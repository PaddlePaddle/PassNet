import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        in_1 += in_0
        tmp_0 = in_1
        tmp_1 = torch.nn.functional.relu(tmp_0, inplace=True)
        tmp_0 = None
        tmp_2 = torch.functional.split(tmp_1, 64, 1)
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_2 = None
        return (tmp_3, tmp_4, tmp_1)