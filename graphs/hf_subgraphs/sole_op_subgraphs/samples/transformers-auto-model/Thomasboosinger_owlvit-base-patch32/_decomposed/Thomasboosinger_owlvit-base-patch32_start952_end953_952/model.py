import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.linalg.norm(in_0, ord=2, dim=-1, keepdim=True)
        return (tmp_0,)