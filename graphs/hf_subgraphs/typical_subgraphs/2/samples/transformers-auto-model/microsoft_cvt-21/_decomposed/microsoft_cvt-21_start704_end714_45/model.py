import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.gelu(in_5, approximate='none')
        tmp_5 = torch.nn.functional.linear(tmp_4, tmp_3, tmp_2)
        tmp_4 = tmp_3 = tmp_2 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_5 = None
        tmp_7 = tmp_6 + in_4
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (384,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_9 = torch.functional.split(tmp_8, [1, 196], 1)
        tmp_8 = None
        tmp_10 = tmp_9[0]
        tmp_11 = tmp_9[1]
        tmp_9 = None
        tmp_12 = tmp_11.permute(0, 2, 1)
        tmp_11 = None
        tmp_13 = tmp_12.view(1, 384, 14, 14)
        tmp_12 = None
        return (tmp_10, tmp_7, tmp_13)