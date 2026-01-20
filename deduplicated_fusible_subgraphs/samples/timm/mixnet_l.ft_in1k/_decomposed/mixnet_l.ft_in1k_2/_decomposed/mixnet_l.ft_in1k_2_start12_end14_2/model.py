import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.conv2d(in_0, w_0, w_1, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = torch.nn.functional.silu(tmp_0, inplace=True)
        tmp_0 = None
        return (tmp_1,)