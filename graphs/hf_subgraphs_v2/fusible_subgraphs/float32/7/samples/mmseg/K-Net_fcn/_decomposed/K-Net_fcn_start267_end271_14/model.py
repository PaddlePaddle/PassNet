import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2[Ellipsis, slice(None, 256, None)]
        tmp_4 = tmp_2[Ellipsis, slice(-256, None, None)]
        tmp_2 = None
        tmp_5 = in_2.unsqueeze(-2)
        return (tmp_3, tmp_4, tmp_5)