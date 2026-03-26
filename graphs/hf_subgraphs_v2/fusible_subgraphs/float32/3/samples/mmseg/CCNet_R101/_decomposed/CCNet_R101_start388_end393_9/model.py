import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
        in_3 += tmp_1
        tmp_2 = in_3
        tmp_1 = None
        tmp_3 = tmp_2 * tmp_0
        tmp_2 = tmp_0 = None
        tmp_4 = tmp_3 + in_2
        tmp_3 = None
        tmp_5 = tmp_4.contiguous()
        tmp_4 = None
        return (tmp_5,)