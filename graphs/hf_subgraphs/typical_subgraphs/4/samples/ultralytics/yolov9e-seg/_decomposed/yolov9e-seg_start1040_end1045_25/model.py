import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_1 = torch.nn.functional.avg_pool2d(tmp_0, 2, 1, 0, False, True)
        tmp_2 = tmp_1.chunk(2, 1)
        tmp_1 = None
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_2 = None
        return (tmp_3, tmp_4, tmp_0)