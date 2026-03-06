import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.linear(input=in_0, weight=tmp_0, bias=tmp_1)
        tmp_0 = tmp_1 = None
        return (tmp_2,)