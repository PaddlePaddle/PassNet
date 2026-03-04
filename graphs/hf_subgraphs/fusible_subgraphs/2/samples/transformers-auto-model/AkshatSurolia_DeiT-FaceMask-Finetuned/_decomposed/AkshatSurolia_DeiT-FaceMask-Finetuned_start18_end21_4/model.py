import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_1.contiguous()
        tmp_1 = in_0.contiguous()
        tmp_2 = in_2.contiguous()
        return (tmp_1, tmp_0, tmp_2)