import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.contiguous()
        tmp_1 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(1, -1, 384, 9)
        tmp_3 = None
        tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
        tmp_4 = None
        return (tmp_5,)