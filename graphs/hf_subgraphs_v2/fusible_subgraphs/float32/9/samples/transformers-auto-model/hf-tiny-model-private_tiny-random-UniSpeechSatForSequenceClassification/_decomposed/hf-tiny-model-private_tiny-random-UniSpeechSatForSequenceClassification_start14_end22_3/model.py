import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.conv1d(in_3, in_5, tmp_2, (1,), (8,), (1,), 2)
        tmp_2 = None
        tmp_4 = tmp_3[slice(None, None, None), slice(None, None, None), slice(None, -1, None)]
        tmp_3 = None
        tmp_5 = torch.nn.functional.gelu(tmp_4)
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = in_4 + tmp_6
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), tmp_1, tmp_0, 1e-05)
        tmp_7 = tmp_1 = tmp_0 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
        tmp_8 = None
        tmp_10 = torch.rand([])
        tmp_10 = None
        return (tmp_9,)