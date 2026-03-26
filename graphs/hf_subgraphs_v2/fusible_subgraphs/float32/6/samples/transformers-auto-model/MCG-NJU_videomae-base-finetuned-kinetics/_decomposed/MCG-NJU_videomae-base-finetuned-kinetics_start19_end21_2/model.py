import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1.contiguous()
        tmp_1 = in_0.contiguous()
        return (tmp_1, tmp_0)