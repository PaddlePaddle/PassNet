import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = torch.cat([tmp_0, in_1, in_2, in_3, in_4], dim=1)
        tmp_0 = None
        return (tmp_1,)