import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3, w_4):
        tmp_0 = torch.conv2d(in_1, w_1, w_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
        tmp_0 = None
        tmp_2 = w_2.unsqueeze(-1)
        tmp_3 = tmp_2.unsqueeze(-1)
        tmp_2 = None
        tmp_4 = tmp_3 * tmp_1
        tmp_3 = tmp_1 = None
        tmp_5 = in_0 + tmp_4
        tmp_4 = None
        tmp_6 = tmp_5.flatten(2)
        tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (256,), w_4, w_3, 1e-06)
        tmp_7 = None
        tmp_9 = tmp_8.view(1, 7, 7, 256)
        tmp_8 = None
        tmp_10 = tmp_9.permute(0, 3, 1, 2)
        tmp_9 = None
        return (tmp_10,)