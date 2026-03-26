import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.cat((in_0, in_1), dim=2)
        tmp_5 = torch.nn.functional.linear(tmp_4, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_6 = tmp_5.transpose(2, 1)
        tmp_5 = None
        tmp_7 = torch.nn.functional.softmax(tmp_6, dim=2)
        tmp_6 = None
        tmp_8 = torch.matmul(tmp_7, tmp_4)
        tmp_7 = tmp_4 = None
        tmp_9 = tmp_8.squeeze(1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_3, tmp_2)
        tmp_9 = tmp_3 = tmp_2 = None
        tmp_11 = torch.nn.functional.softmax(tmp_10, dim=1)
        tmp_10 = None
        tmp_12 = torch.linspace(0, 4, steps=5, device=device(type='cuda', index=0))
        tmp_13 = tmp_11 * tmp_12
        tmp_11 = tmp_12 = None
        tmp_14 = tmp_13.sum(dim=1)
        tmp_13 = None
        tmp_15 = 5 - tmp_14
        tmp_14 = None
        return (tmp_15,)