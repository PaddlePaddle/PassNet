import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.sigmoid(in_3)
        tmp_1 = torch.functional.split(tmp_0, [20, 40, 80], dim=1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_4 = tmp_1[2]
        tmp_1 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_2, size=(64, 48), mode='nearest')
        tmp_2 = None
        tmp_6 = in_0 * tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.interpolate(tmp_3, size=(32, 24), mode='nearest')
        tmp_3 = None
        tmp_8 = in_1 * tmp_7
        tmp_7 = None
        tmp_9 = torch.nn.functional.interpolate(tmp_4, size=(16, 12), mode='nearest')
        tmp_4 = None
        tmp_10 = in_2 * tmp_9
        tmp_9 = None
        return (tmp_6, tmp_8, tmp_10)