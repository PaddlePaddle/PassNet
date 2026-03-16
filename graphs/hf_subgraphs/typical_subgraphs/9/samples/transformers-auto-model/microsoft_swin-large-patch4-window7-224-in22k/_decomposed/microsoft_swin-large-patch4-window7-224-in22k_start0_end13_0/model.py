import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = torch.conv2d(tmp_0, tmp_4, tmp_3, (4, 4), (0, 0), (1, 1), 1)
        tmp_0 = tmp_4 = tmp_3 = None
        tmp_8 = tmp_7.flatten(2)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (192,), tmp_2, tmp_1, 1e-05)
        tmp_9 = tmp_2 = tmp_1 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (192,), tmp_6, tmp_5, 1e-05)
        tmp_6 = tmp_5 = None
        tmp_13 = tmp_12.view(1, 56, 56, 192)
        tmp_12 = None
        tmp_14 = torch.nn.functional.pad(tmp_13, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_13 = None
        tmp_15 = tmp_14.view(1, 8, 7, 8, 7, 192)
        tmp_14 = None
        tmp_16 = tmp_15.permute(0, 1, 3, 2, 4, 5)
        tmp_15 = None
        tmp_17 = tmp_16.contiguous()
        tmp_16 = None
        tmp_18 = tmp_17.view(-1, 7, 7, 192)
        tmp_17 = None
        tmp_19 = tmp_18.view(-1, 49, 192)
        tmp_18 = None
        return (tmp_11, tmp_19)