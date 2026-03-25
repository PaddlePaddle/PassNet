import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2.contiguous()
        tmp_3 = tmp_1.view((1, 1, 32))
        tmp_1 = None
        tmp_4 = tmp_2.unsqueeze(2)
        tmp_5 = tmp_4.expand((1, 4096, 32, 512))
        tmp_4 = None
        tmp_6 = tmp_0.view((1, 1, 32, 512))
        tmp_0 = None
        return (tmp_5, tmp_6, tmp_3, tmp_2)