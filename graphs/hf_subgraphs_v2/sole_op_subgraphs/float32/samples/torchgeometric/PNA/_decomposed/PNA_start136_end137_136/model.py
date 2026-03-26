import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 <= 0.0031622776601683794
        return (tmp_0,)