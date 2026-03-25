import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = torch.conv2d(tmp_0, tmp_4, tmp_3, (4, 4), (0, 0), (1, 1), 1)
        tmp_0 = tmp_4 = tmp_3 = None
        tmp_6 = tmp_5.flatten(2)
        tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (96,), tmp_2, tmp_1, 1e-05)
        tmp_7 = tmp_2 = tmp_1 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        tmp_10 = tmp_9.view(1, 256, 256, 96)
        tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_10 = None
        tmp_12 = tmp_11.view(1, 32, 8, 32, 8, 96)
        tmp_11 = None
        tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
        tmp_12 = None
        tmp_14 = tmp_13.contiguous()
        tmp_13 = None
        tmp_15 = tmp_14.view(-1, 8, 8, 96)
        tmp_14 = None
        tmp_16 = tmp_15.view(-1, 64, 96)
        tmp_15 = None
        return (tmp_9, tmp_16)