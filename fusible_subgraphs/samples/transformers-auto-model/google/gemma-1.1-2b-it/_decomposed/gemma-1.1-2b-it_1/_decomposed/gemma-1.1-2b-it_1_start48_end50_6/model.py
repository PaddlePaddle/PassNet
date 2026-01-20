import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = torch.nn.functional.linear(in_0, w_0, None)
        tmp_1 = in_1 * tmp_0
        tmp_0 = None
        return (tmp_1,)