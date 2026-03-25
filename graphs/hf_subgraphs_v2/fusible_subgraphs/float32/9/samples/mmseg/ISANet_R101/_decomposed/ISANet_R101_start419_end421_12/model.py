import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1.reshape(1, 512, 64, 64)
        tmp_1 = torch.cat([tmp_0, in_0], dim=1)
        tmp_0 = None
        return (tmp_1,)