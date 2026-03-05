import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.new_zeros((1000, 32))
        return (tmp_0,)