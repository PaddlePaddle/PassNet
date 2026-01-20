import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0):
        tmp_0 = w_0[in_2]
        tmp_1 = tmp_0.view(4, 4, -1)
        tmp_0 = None
        tmp_2 = tmp_1.permute(2, 0, 1)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = tmp_3.unsqueeze(0)
        tmp_3 = None
        tmp_5 = in_1 + tmp_4
        tmp_4 = None
        tmp_6 = tmp_5.view(1, 16, 4, 4, 4)
        tmp_5 = None
        tmp_7 = in_0.unsqueeze(1)
        tmp_8 = tmp_7.unsqueeze(0)
        tmp_7 = None
        tmp_9 = tmp_6 + tmp_8
        tmp_6 = tmp_8 = None
        tmp_10 = tmp_9.view(-1, 4, 4, 4)
        tmp_9 = None
        return (tmp_10,)