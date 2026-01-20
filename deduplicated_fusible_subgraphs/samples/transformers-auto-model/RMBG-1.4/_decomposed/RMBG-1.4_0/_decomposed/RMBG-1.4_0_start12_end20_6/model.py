import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, w_0, w_1):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_1 = torch.nn.functional.interpolate(tmp_0, size=(640, 640), mode='bilinear')
        tmp_0 = None
        tmp_2 = torch.nn.functional.sigmoid(in_1)
        tmp_3 = torch.nn.functional.sigmoid(in_2)
        tmp_4 = torch.nn.functional.sigmoid(in_3)
        tmp_5 = torch.nn.functional.sigmoid(in_4)
        tmp_6 = torch.nn.functional.sigmoid(in_5)
        tmp_7 = torch.nn.functional.sigmoid(tmp_1)
        tmp_1 = None
        return (tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7)