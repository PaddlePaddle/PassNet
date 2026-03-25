import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = in_0
        tmp_2 = torch.nn.functional.linear(tmp_1, tmp_0, None)
        tmp_1 = tmp_0 = None
        return (tmp_2,)