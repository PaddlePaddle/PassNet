import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = in_1.transpose(1, 2)
        tmp_7 = tmp_6.view(1, 1024, 16, 16)
        tmp_6 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1024)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = tmp_8.flatten(2)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = torch.nn.functional.gelu(tmp_10)
        tmp_10 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
        tmp_11 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_1, tmp_0)
        tmp_12 = tmp_1 = tmp_0 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False)
        tmp_13 = None
        tmp_15 = tmp_14 + in_0
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (256,), tmp_5, tmp_4, 1e-05)
        tmp_15 = tmp_5 = tmp_4 = None
        tmp_17 = tmp_16.reshape(1, 16, 16, -1)
        tmp_16 = None
        tmp_18 = tmp_17.permute(0, 3, 1, 2)
        tmp_17 = None
        tmp_19 = tmp_18.contiguous()
        tmp_18 = None
        return (tmp_19,)