import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_3 * in_0
        tmp_1 = torch.sum(tmp_0, dim=1)
        tmp_0 = None
        tmp_2 = tmp_1.unsqueeze(1)
        tmp_1 = None
        tmp_3 = torch.sigmoid(tmp_2)
        tmp_2 = None
        tmp_4 = torch.nn.functional.interpolate(in_2, size=(64, 64), mode='bilinear', align_corners=False)
        tmp_5 = tmp_3 * tmp_4
        tmp_4 = None
        tmp_6 = 1 - tmp_3
        tmp_3 = None
        tmp_7 = tmp_6 * in_1
        tmp_6 = None
        tmp_8 = tmp_5 + tmp_7
        tmp_5 = tmp_7 = None
        return (tmp_8,)