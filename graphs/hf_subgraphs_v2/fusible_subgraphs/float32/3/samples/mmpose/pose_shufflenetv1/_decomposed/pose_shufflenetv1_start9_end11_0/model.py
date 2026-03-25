import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(32, 3, 20, 28, 28)
        tmp_1 = torch.transpose(tmp_0, 1, 2)
        tmp_0 = None
        return (tmp_1,)