import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = torch.nn.functional.linear(input=in_1, weight=tmp_0, bias=in_0)
        tmp_0 = None
        return (tmp_1,)