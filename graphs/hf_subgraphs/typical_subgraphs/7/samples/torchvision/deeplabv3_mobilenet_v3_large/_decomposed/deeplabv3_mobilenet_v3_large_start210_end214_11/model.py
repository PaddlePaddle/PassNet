import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
        tmp_2 = None
        tmp_4 = torch.conv2d(tmp_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_1 = tmp_0 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(224, 224), mode='bilinear', align_corners=False)
        tmp_4 = None
        return (tmp_5,)