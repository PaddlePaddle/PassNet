import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.nn.functional.gelu(in_6, approximate='none')
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_1, tmp_0)
        tmp_7 = tmp_1 = tmp_0 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        tmp_10 = in_7 + tmp_9
        tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (384,), tmp_3, tmp_2, 1e-06)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = tmp_11[slice(None, None, None), slice(0, None, None)]
        tmp_11 = None
        tmp_13 = tmp_12.reshape(32, 16, 12, -1)
        tmp_12 = None
        tmp_14 = tmp_13.permute(0, 3, 1, 2)
        tmp_13 = None
        tmp_15 = torch.nn.functional.relu(tmp_14)
        tmp_14 = None
        tmp_16 = torch.nn.functional.interpolate(tmp_15, None, 4.0, 'bilinear', False)
        tmp_15 = None
        tmp_17 = torch.conv2d(tmp_16, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 1)
        tmp_16 = tmp_5 = tmp_4 = None
        return (tmp_17,)