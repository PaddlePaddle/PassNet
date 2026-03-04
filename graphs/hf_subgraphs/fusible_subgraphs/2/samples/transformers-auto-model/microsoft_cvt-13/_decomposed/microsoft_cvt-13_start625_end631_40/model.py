import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 + in_0
        tmp_1 = torch.functional.split(tmp_0, [1, 196], 1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        tmp_4 = tmp_3.permute(0, 2, 1)
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 384, 14, 14)
        tmp_4 = None
        return (tmp_2, tmp_5)