import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6):
        tmp_0 = torch.conv2d(in_0, w_4, None, (14, 14), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0.flatten(2)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = w_5.expand(1, -1, -1)
        tmp_4 = torch.cat([tmp_3, tmp_2], dim=1)
        tmp_3 = tmp_2 = None
        tmp_5 = tmp_4 + w_6
        tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (1024,), w_3, w_2, 1e-05)
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (1024,), w_1, w_0, 1e-05)
        return (tmp_7, tmp_8)