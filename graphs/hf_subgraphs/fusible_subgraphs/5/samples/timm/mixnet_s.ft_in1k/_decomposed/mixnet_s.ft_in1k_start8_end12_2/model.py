import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 + in_0
        tmp_1 = torch.functional.split(tmp_0, [8, 8], 1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        return (tmp_2, tmp_3)