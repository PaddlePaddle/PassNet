import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0[slice(1024, 2048, None)]
        return (tmp_0,)