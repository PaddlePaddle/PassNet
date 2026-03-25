import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.norm(p=2, dim=-1, keepdim=True)
        return (tmp_0,)