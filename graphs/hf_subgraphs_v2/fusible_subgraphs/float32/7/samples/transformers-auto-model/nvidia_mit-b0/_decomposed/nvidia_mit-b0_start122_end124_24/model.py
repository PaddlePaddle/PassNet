import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.transpose(1, 2)
        tmp_1 = tmp_0.view(32, 256, 64, 64)
        tmp_0 = None
        return (tmp_1,)