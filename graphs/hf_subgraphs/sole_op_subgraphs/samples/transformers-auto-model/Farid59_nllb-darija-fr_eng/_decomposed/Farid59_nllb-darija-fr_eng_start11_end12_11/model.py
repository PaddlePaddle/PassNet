import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.to(torch.bfloat16)
        return (tmp_0,)