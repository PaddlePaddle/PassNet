import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.new_zeros((16, 16))
        return (tmp_0,)