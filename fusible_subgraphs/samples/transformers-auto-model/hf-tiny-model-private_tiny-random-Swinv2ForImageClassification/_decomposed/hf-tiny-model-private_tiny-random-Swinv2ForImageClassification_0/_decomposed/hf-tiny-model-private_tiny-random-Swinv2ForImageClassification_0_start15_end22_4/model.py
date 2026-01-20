import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0[in_1]
        tmp_1 = tmp_0.view(4, 4, -1)
        tmp_0 = None
        tmp_2 = tmp_1.permute(2, 0, 1)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = torch.sigmoid(tmp_3)
        tmp_3 = None
        tmp_5 = 16 * tmp_4
        tmp_4 = None
        tmp_6 = tmp_5.unsqueeze(0)
        tmp_5 = None
        return (tmp_6,)