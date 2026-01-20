import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_1, w_1, w_0)
        tmp_1 = in_0[Ellipsis, None]
        return (tmp_0, tmp_1)