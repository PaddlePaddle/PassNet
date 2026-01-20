import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.softmax(in_1, -1)
        tmp_1 = tmp_0.view(1, 625, 8, 1, 4)
        tmp_0 = None
        tmp_2 = in_0[Ellipsis, 1]
        tmp_3 = in_0[Ellipsis, 0]
        return (tmp_1, tmp_2, tmp_3)