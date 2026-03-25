import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.conv1d(in_3, in_4, tmp_2, (2,), (15,), (1,), 16)
        tmp_2 = None
        tmp_4 = torch.nn.functional.gelu(tmp_3)
        tmp_3 = None
        tmp_5 = torch.avg_pool1d(in_3, (2,), (2,), (0,), False, True)
        tmp_6 = tmp_5[Ellipsis, slice(None, 124, None)]
        tmp_5 = None
        tmp_7 = tmp_4[Ellipsis, slice(None, 124, None)]
        tmp_4 = None
        tmp_8 = tmp_6 + tmp_7
        tmp_6 = tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), tmp_1, tmp_0, 1e-05)
        tmp_9 = tmp_1 = tmp_0 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
        tmp_10 = None
        tmp_12 = torch.rand([])
        tmp_12 = None
        return (tmp_11,)