import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        in_2 -= in_0
        tmp_0 = in_2
        tmp_1 = in_1.mean(dim=-1, keepdim=True)
        return (tmp_1, tmp_0)