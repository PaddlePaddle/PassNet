import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_2.softmax(dim=-1)
        tmp_2 = tmp_0.view(1, -1, 1, 1)
        tmp_0 = None
        tmp_3 = torch.sigmoid(tmp_2)
        tmp_4 = 1.0 - tmp_3
        tmp_3 = None
        tmp_5 = tmp_4 * in_1
        tmp_4 = None
        tmp_6 = torch.sigmoid(tmp_2)
        tmp_2 = None
        tmp_7 = tmp_6 * tmp_1
        tmp_6 = tmp_1 = None
        tmp_8 = tmp_5 + tmp_7
        tmp_5 = tmp_7 = None
        return (tmp_8,)