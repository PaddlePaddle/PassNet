import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.contiguous()
        tmp_1 = tmp_0.view(-1, 2, 2, 32)
        tmp_0 = None
        tmp_2 = tmp_1.view(-1, 4, 32)
        tmp_1 = None
        return (tmp_2,)