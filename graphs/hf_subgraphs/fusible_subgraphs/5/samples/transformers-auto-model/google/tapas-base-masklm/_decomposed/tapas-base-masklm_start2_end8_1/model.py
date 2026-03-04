import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.arange(start=0, end=in_2, device=device(type='cuda'))
        tmp_3 = tmp_2 * tmp_1
        tmp_2 = tmp_1 = None
        tmp_4 = tmp_3.view((1,))
        tmp_3 = None
        tmp_5 = tmp_4.unsqueeze(-1)
        tmp_4 = None
        tmp_6 = tmp_5 + tmp_0
        tmp_5 = tmp_0 = None
        tmp_7 = tmp_6.view(-1)
        tmp_6 = None
        return (tmp_7,)