import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = torch.conv2d(in_0, w_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = torch.nn.functional.pad(tmp_0, [2, 2, 2, 2], 'constant', None)
        tmp_0 = None
        tmp_2 = tmp_1.unfold(2, 12, 8)
        tmp_1 = None
        return (tmp_2,)