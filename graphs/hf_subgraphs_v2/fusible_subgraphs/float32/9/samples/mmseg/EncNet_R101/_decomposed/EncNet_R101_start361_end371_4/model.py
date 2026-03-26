import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1 - in_2
        tmp_2 = tmp_1.pow(2)
        tmp_1 = None
        tmp_3 = tmp_2.sum(dim=3)
        tmp_2 = None
        tmp_4 = in_3 * tmp_3
        tmp_3 = None
        tmp_5 = torch.nn.functional.softmax(tmp_4, dim=2)
        tmp_4 = None
        tmp_6 = tmp_0.view((1, 1, 32, 512))
        tmp_0 = None
        tmp_7 = in_4.unsqueeze(2)
        tmp_8 = tmp_7.expand((1, 4096, 32, 512))
        tmp_7 = None
        tmp_9 = tmp_5.unsqueeze(3)
        tmp_5 = None
        tmp_10 = tmp_8 - tmp_6
        tmp_8 = tmp_6 = None
        return (tmp_10, tmp_9)