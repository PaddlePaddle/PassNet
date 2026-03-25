import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.pad(tmp_1, [2, 2, 2, 2], 'constant', None)
        tmp_1 = None
        tmp_3 = tmp_2.unfold(2, 12, 8)
        tmp_2 = None
        return (tmp_3,)