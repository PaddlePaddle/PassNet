import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1.softmax(dim=-1)
        tmp_2 = tmp_0.view(1, -1, 1, 1)
        tmp_0 = None
        return (tmp_2, tmp_1)