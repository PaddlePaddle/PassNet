import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.cat((in_0, in_1, in_2), dim=1)
        return (tmp_0,)