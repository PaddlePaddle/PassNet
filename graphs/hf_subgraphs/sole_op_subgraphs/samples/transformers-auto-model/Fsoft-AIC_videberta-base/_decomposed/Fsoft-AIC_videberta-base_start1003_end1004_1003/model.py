import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.gather(in_1, dim=-1, index=in_0)
        return (tmp_0,)