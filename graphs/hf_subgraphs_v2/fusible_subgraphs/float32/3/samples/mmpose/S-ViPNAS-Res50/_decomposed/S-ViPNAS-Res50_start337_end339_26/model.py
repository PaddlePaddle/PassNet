import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.matmul(in_1, in_0)
        tmp_1 = tmp_0.view(16, 608, 1, 1)
        tmp_0 = None
        return (tmp_1,)