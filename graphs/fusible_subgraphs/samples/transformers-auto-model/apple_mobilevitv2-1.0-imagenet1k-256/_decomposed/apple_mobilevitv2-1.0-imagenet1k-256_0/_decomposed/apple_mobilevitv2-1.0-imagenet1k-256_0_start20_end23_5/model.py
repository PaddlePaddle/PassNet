import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = torch.nn.functional.silu(tmp_0, inplace=False)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
        tmp_1 = None
        return (tmp_2,)