import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.sigmoid(in_2)
        tmp_1 = torch.functional.split(tmp_0, [20, 40], dim=1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        tmp_4 = torch.nn.functional.interpolate(tmp_2, size=(64, 48), mode='nearest')
        tmp_2 = None
        tmp_5 = in_0 * tmp_4
        tmp_4 = None
        tmp_6 = torch.nn.functional.interpolate(tmp_3, size=(32, 24), mode='nearest')
        tmp_3 = None
        tmp_7 = in_1 * tmp_6
        tmp_6 = None
        return (tmp_5, tmp_7)