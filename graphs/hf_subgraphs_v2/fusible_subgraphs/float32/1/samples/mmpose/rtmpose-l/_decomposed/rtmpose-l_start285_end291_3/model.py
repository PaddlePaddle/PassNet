import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_2 * in_1
        tmp_2 = tmp_1 + tmp_0
        tmp_1 = tmp_0 = None
        tmp_3 = torch.unbind(tmp_2, dim=2)
        tmp_2 = None
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_3 = None
        tmp_6 = tmp_5.permute(0, 2, 1)
        tmp_5 = None
        return (tmp_6, tmp_4)