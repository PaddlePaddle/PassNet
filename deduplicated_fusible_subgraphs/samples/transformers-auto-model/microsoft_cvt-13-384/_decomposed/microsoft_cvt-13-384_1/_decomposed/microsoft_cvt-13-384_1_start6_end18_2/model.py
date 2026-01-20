import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = torch.conv2d(in_0, w_3, w_2, (2, 2), (1, 1), (1, 1), 1)
        tmp_1 = tmp_0.view(1, 192, 2304)
        tmp_0 = None
        tmp_2 = tmp_1.permute(0, 2, 1)
        tmp_1 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (192,), w_1, w_0, 1e-05)
        tmp_2 = None
        tmp_4 = tmp_3.permute(0, 2, 1)
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 192, 48, 48)
        tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_5 = None
        tmp_7 = tmp_6.view(1, 192, 2304)
        tmp_6 = None
        tmp_8 = tmp_7.permute(0, 2, 1)
        tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (192,), w_5, w_4, 1e-05)
        tmp_10 = tmp_9.permute(0, 2, 1)
        tmp_9 = None
        tmp_11 = tmp_10.view(1, 192, 48, 48)
        tmp_10 = None
        return (tmp_8, tmp_11)