import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 * 0.3535533905932738
        return (tmp_0,)