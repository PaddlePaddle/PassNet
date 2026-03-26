import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.matmul(in_1, in_0)
        tmp_1 = tmp_0.view(45, 1, 4, 45)
        tmp_0 = None
        tmp_2 = tmp_1.permute(1, 2, 0, 3)
        tmp_1 = None
        return (tmp_2,)