import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = torch.cat([in_0, in_1, in_2, in_3, in_4, in_5, in_6], dim=1)
        return (tmp_0,)