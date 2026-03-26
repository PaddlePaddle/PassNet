import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.cat([in_0, in_1, in_2, in_3], -1)
        tmp_1 = tmp_0.view(1, -1, 2048)
        tmp_0 = None
        return (tmp_1,)