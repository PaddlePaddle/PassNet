import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.bmm(in_0, in_1)
        tmp_1 = tmp_0.view(1, 8, 2500, 32)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        return (tmp_2,)