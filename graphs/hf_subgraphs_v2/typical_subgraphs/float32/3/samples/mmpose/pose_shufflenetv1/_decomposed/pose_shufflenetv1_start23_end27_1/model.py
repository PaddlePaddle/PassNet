import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(32, 3, 20, 28, 28)
        tmp_1 = torch.transpose(tmp_0, 1, 2)
        tmp_0 = None
        tmp_2 = tmp_1.contiguous()
        tmp_1 = None
        tmp_3 = tmp_2.view(32, 60, 28, 28)
        tmp_2 = None
        return (tmp_3,)