import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.special.logit(in_0, eps=1e-05)
        return (tmp_0,)