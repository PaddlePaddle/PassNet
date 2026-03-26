import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(2, 2), mode='bilinear', align_corners=False)
        tmp_1 = tmp_0.reshape(240, 2, 1, 2)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = tmp_2.reshape(1, 240, 1, 4)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 3)
        tmp_3 = None
        tmp_5 = tmp_4.reshape(4, 1, -1)
        tmp_4 = None
        return (tmp_5,)