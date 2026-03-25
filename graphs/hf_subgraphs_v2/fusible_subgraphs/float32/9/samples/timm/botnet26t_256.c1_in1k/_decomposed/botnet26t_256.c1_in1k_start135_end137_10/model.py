import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 @ in_1
        tmp_1 = tmp_0.reshape(-1, 16, 31)
        tmp_0 = None
        return (tmp_1,)