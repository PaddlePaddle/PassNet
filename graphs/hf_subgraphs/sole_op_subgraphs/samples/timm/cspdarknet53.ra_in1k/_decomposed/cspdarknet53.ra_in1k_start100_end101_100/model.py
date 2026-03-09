import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.leaky_relu(in_0, 0.01, True)
        return (tmp_0,)