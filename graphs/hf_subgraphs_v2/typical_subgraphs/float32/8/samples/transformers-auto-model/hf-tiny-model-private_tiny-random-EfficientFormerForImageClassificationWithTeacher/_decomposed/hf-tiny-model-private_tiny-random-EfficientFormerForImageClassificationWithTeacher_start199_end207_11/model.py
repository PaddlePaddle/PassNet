import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
        tmp_3 = tmp_2 - in_2
        tmp_2 = None
        tmp_4 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_5 = tmp_4.unsqueeze(-1)
        tmp_4 = None
        tmp_6 = tmp_5 * tmp_3
        tmp_5 = tmp_3 = None
        tmp_7 = in_2 + tmp_6
        tmp_6 = None
        tmp_8 = tmp_1.unsqueeze(-1)
        tmp_1 = None
        tmp_9 = tmp_8.unsqueeze(-1)
        tmp_8 = None
        return (tmp_7, tmp_9)