import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.sigmoid(in_0)
        tmp_1 = tmp_0.view(128, 64, 1, 1)
        tmp_0 = None
        return (tmp_1,)