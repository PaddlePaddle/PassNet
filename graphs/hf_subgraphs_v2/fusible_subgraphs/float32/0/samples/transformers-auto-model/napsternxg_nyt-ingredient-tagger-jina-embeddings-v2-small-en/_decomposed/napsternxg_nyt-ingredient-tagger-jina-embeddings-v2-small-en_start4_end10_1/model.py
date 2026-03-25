import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = torch.nn.functional.embedding(tmp_0, tmp_4, 0, None, 2.0, False, False)
        tmp_0 = tmp_4 = None
        tmp_8 = torch.nn.functional.embedding(tmp_6, tmp_3, None, None, 2.0, False, False)
        tmp_6 = tmp_3 = None
        tmp_9 = tmp_7 + tmp_8
        tmp_7 = tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (512,), tmp_2, tmp_1, 1e-12)
        tmp_9 = tmp_2 = tmp_1 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
        tmp_10 = None
        tmp_12 = tmp_5[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, 64, None)]
        tmp_5 = None
        return (tmp_11, tmp_12)