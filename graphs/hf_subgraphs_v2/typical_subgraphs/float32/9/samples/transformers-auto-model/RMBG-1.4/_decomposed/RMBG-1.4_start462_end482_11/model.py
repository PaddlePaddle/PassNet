import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = w_10
        tmp_11 = w_11
        tmp_12 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_13 = tmp_12 + in_6
        tmp_12 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_15 = torch.nn.functional.interpolate(tmp_14, size=(640, 640), mode='bilinear')
        tmp_14 = None
        tmp_16 = torch.conv2d(in_1, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_17 = torch.nn.functional.interpolate(tmp_16, size=(640, 640), mode='bilinear')
        tmp_16 = None
        tmp_18 = torch.conv2d(in_2, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_19 = torch.nn.functional.interpolate(tmp_18, size=(640, 640), mode='bilinear')
        tmp_18 = None
        tmp_20 = torch.conv2d(in_3, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_21 = torch.nn.functional.interpolate(tmp_20, size=(640, 640), mode='bilinear')
        tmp_20 = None
        tmp_22 = torch.conv2d(in_4, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_9 = tmp_8 = None
        tmp_23 = torch.nn.functional.interpolate(tmp_22, size=(640, 640), mode='bilinear')
        tmp_22 = None
        tmp_24 = torch.conv2d(in_5, tmp_11, tmp_10, (1, 1), (1, 1), (1, 1), 1)
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