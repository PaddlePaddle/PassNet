import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.relu(in_1)
        tmp_2 = torch.cat([tmp_0, tmp_1], axis=1)
        tmp_0 = tmp_1 = None
        return (tmp_2,)