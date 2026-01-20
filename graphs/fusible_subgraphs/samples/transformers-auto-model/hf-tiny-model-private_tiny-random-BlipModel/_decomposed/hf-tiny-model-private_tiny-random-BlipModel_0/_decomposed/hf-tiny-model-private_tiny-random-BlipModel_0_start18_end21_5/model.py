import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.matmul(in_0, in_2)
        tmp_1 = tmp_0 * in_1
        tmp_0 = None
        tmp_2 = tmp_1.t()
        return (tmp_1, tmp_2)