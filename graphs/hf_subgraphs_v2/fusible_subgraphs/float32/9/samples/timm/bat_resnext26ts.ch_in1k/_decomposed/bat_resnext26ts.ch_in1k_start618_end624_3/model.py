import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.contiguous()
        tmp_1 = tmp_0.view(1, 128, 8, 8)
        tmp_0 = None
        tmp_2 = in_1.view(1, 2, 1, 8, 8)
        tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 128, 8, 8)
        tmp_4 = None
        return (tmp_1, tmp_5)