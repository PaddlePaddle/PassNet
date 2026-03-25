import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
        tmp_2 = torch.cat([tmp_0, tmp_1], dim=-1)
        tmp_0 = tmp_1 = None
        tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
        tmp_2 = None
        tmp_4 = tmp_3[Ellipsis, slice(None, 64, None)]
        return (tmp_3, tmp_4)