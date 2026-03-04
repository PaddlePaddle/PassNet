import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.leaky_relu(in_0, 0.01, True)
        tmp_1 = tmp_0.split(256, dim=1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        return (tmp_3, tmp_2)