import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 @ in_0
        tmp_1 = tmp_0.view(8, 384, 20, 20)
        tmp_0 = None
        return (tmp_1,)