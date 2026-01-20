import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = tmp_0.view(1, -1, 4, 8)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        return (tmp_2,)