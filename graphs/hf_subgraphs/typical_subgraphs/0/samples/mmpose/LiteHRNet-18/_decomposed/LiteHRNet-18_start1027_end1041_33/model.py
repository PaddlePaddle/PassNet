import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = torch.sigmoid(in_4)
        tmp_1 = torch.functional.split(tmp_0, [20, 40, 80, 160], dim=1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_4 = tmp_1[2]
        tmp_5 = tmp_1[3]
        tmp_1 = None
        tmp_6 = torch.nn.functional.interpolate(tmp_2, size=(64, 48), mode='nearest')
        tmp_2 = None
        tmp_7 = in_0 * tmp_6
        tmp_6 = None
        tmp_8 = torch.nn.functional.interpolate(tmp_3, size=(32, 24), mode='nearest')
        tmp_3 = None
        tmp_9 = in_1 * tmp_8
        tmp_8 = None
        tmp_10 = torch.nn.functional.interpolate(tmp_4, size=(16, 12), mode='nearest')
        tmp_4 = None
        tmp_11 = in_2 * tmp_10
        tmp_10 = None
        tmp_12 = torch.nn.functional.interpolate(tmp_5, size=(8, 6), mode='nearest')
        tmp_5 = None
        tmp_13 = in_3 * tmp_12
        tmp_12 = None
        return (tmp_7, tmp_9, tmp_11, tmp_13)