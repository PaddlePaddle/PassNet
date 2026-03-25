import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.matmul(in_1, in_0)
        tmp_1 = torch.reshape(tmp_0, [-1, 16])
        tmp_0 = None
        tmp_2 = in_2.transpose(-1, -2)
        return (tmp_1, tmp_2)