import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2):
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
        tmp_10 = in_2.transpose(1, 2)
        tmp_11 = tmp_10.view(1, 2048, 12, 12)
        tmp_10 = None
        tmp_12 = torch.conv2d(tmp_11, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 2048)
        tmp_11 = tmp_7 = tmp_6 = None
        tmp_13 = tmp_12.flatten(2)
        tmp_12 = None
        tmp_14 = tmp_13.transpose(1, 2)
        tmp_13 = None
        tmp_15 = torch.nn.functional.gelu(tmp_14)
        tmp_14 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.0, False, False)
        tmp_15 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_5, tmp_4)
        tmp_16 = tmp_5 = tmp_4 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, 0.0, False, False)
        tmp_17 = None
        tmp_19 = tmp_18 + in_1
        tmp_18 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (512,), tmp_9, tmp_8, 1e-05)
        tmp_19 = tmp_9 = tmp_8 = None
        tmp_21 = tmp_20.reshape(1, 12, 12, -1)
        tmp_20 = None
        tmp_22 = tmp_21.permute(0, 3, 1, 2)
        tmp_21 = None
        tmp_23 = tmp_22.contiguous()
        tmp_22 = None
        tmp_24 = torch.conv2d(tmp_23, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_23 = tmp_1 = tmp_0 = None
        tmp_25 = torch.nn.functional.interpolate(tmp_24, None, 2.0, 'bilinear', False, recompute_scale_factor=None)
        tmp_24 = None
        tmp_26 = torch.conv2d(in_0, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_27 = torch.cat((tmp_26, tmp_25), dim=1)
        return (tmp_27, tmp_25, tmp_26)