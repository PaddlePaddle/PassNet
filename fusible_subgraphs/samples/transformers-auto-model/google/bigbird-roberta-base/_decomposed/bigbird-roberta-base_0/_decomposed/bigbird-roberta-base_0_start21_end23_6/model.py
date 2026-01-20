import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1.contiguous()
        tmp_1 = in_0.squeeze(-1)
        return (tmp_0, tmp_1)