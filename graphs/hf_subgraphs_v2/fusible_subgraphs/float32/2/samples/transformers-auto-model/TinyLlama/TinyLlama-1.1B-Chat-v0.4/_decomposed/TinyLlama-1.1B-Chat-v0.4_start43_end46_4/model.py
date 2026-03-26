import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
        tmp_0 = None
        tmp_2 = tmp_1.view((16, 128, -1, 64))
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        return (tmp_3,)