import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, size=(640, 640), mode='bilinear')
        tmp_2 = None
        tmp_4 = torch.nn.functional.sigmoid(in_3)
        tmp_5 = torch.nn.functional.sigmoid(in_4)
        tmp_6 = torch.nn.functional.sigmoid(in_5)
        tmp_7 = torch.nn.functional.sigmoid(in_6)
        tmp_8 = torch.nn.functional.sigmoid(in_7)
        tmp_9 = torch.nn.functional.sigmoid(tmp_3)
        tmp_3 = None
        return (tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9)