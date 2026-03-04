import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.tanh(in_1)
        tmp_0 = None
        tmp_1 = in_0[slice(None, None, None), 0]
        return (tmp_1,)