import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_2.contiguous()
        tmp_1 = in_0.view(1, 400, 8, 32)
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = tmp_3.view(8, -1, 32)
        tmp_3 = None
        tmp_5 = in_1.view(8, -1, 32)
        tmp_6 = tmp_0.view(8, -1, 32)
        tmp_0 = None
        tmp_7 = tmp_5.transpose(1, 2)
        tmp_5 = None
        return (tmp_4, tmp_6, tmp_7)