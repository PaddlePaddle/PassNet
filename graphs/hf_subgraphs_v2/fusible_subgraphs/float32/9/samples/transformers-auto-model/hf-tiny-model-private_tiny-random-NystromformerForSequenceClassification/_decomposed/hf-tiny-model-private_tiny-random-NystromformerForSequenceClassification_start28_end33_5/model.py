import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_2, tmp_0, None, (1, 1), (32, 0), (1, 1), 4)
        tmp_0 = None
        in_1 += tmp_1
        tmp_2 = in_1
        tmp_1 = None
        tmp_3 = tmp_2.permute(0, 2, 1, 3)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 12, 32)
        tmp_4 = None
        return (tmp_5,)