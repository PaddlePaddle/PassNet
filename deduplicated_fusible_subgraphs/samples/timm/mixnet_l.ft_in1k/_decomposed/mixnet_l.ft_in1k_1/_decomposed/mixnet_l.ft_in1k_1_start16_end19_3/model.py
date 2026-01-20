import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = torch.conv2d(in_1, w_1, w_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = torch.sigmoid(tmp_0)
        tmp_0 = None
        tmp_2 = in_0 * tmp_1
        tmp_1 = None
        return (tmp_2,)