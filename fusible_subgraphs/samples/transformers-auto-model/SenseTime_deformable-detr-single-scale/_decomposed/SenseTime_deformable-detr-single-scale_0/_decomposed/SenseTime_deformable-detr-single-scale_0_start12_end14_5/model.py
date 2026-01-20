import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = tmp_0.view(1, 625, 8, 1, 4, 2)
        tmp_0 = None
        return (tmp_1,)