import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.cat((in_3, in_0, in_1, in_2), 1)
        return (tmp_0,)