import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = torch.nn.functional.relu(in_12, inplace=True)
        tmp_13 = tmp_12 + in_18
        tmp_12 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_15 = torch.nn.functional.interpolate(tmp_14, size=(640, 640), mode='bilinear')
        tmp_14 = None
        tmp_16 = torch.conv2d(in_13, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_17 = torch.nn.functional.interpolate(tmp_16, size=(640, 640), mode='bilinear')
        tmp_16 = None
        tmp_18 = torch.conv2d(in_14, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_19 = torch.nn.functional.interpolate(tmp_18, size=(640, 640), mode='bilinear')
        tmp_18 = None
        tmp_20 = torch.conv2d(in_15, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_21 = torch.nn.functional.interpolate(tmp_20, size=(640, 640), mode='bilinear')
        tmp_20 = None
        tmp_22 = torch.conv2d(in_16, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_9 = tmp_8 = None
        tmp_23 = torch.nn.functional.interpolate(tmp_22, size=(640, 640), mode='bilinear')
        tmp_22 = None
        tmp_24 = torch.conv2d(in_17, tmp_11, tmp_10, (1, 1), (1, 1), (1, 1), 1)
        tmp_11 = tmp_10 = None
        tmp_25 = torch.nn.functional.interpolate(tmp_24, size=(640, 640), mode='bilinear')
        tmp_24 = None
        tmp_26 = torch.nn.functional.sigmoid(tmp_15)
        tmp_15 = None
        tmp_27 = torch.nn.functional.sigmoid(tmp_17)
        tmp_17 = None
        tmp_28 = torch.nn.functional.sigmoid(tmp_19)
        tmp_19 = None
        tmp_29 = torch.nn.functional.sigmoid(tmp_21)
        tmp_21 = None
        tmp_30 = torch.nn.functional.sigmoid(tmp_23)
        tmp_23 = None
        tmp_31 = torch.nn.functional.sigmoid(tmp_25)
        tmp_25 = None
        return (tmp_13, tmp_26, tmp_27, tmp_28, tmp_29, tmp_30, tmp_31)