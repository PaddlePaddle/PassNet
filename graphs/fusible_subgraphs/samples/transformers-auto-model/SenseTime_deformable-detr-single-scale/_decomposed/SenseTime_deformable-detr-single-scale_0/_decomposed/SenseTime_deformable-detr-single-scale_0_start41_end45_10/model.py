import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.sum(-1)
        tmp_1 = tmp_0.view(1, 256, 625)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        return (tmp_3,)