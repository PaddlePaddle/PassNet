import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.sym_int(in_0)
        return (tmp_0,)