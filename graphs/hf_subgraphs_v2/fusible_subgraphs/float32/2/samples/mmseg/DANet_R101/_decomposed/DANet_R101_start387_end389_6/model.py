import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.bmm(in_0, in_1)
        tmp_1 = tmp_0.view(4, 512, 64, 64)
        tmp_0 = None
        return (tmp_1,)