import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.conv2d(in_0, tmp_1, tmp_0, (1, 1), 'same', (1, 1), 1536)
        tmp_1 = tmp_0 = None
        return (tmp_2,)