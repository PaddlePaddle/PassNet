import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_3 = torch.nn.functional.interpolate(tmp_2, [128, 128], None, 'bilinear', False)
        tmp_2 = None
        tmp_4 = torch.nn.functional.interpolate(tmp_3, (128, 128), None, 'bilinear', False)
        tmp_3 = None
        tmp_5 = in_0 + tmp_4
        tmp_4 = None
        tmp_6 = torch.nn.functional.dropout2d(tmp_5, 0.1, False, False)
        tmp_5 = None
        tmp_7 = torch.conv2d(tmp_6, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_1 = tmp_0 = None
        return (tmp_7,)