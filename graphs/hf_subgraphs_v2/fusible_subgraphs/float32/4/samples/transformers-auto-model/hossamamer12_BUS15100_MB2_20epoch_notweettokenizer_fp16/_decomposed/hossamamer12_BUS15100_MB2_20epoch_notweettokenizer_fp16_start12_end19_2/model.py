import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.nn.functional.embedding(in_5, tmp_2, None, None, 2.0, False, False)
        tmp_2 = None
        tmp_6 = torch.nn.functional.embedding(tmp_4, tmp_3, None, None, 2.0, False, False)
        tmp_4 = tmp_3 = None
        tmp_7 = in_6 + tmp_5
        tmp_5 = None
        tmp_8 = tmp_7 + tmp_6
        tmp_7 = tmp_6 = None
        tmp_9 = tmp_8 * tmp_1
        tmp_8 = tmp_1 = None
        tmp_10 = tmp_9 + tmp_0
        tmp_9 = tmp_0 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        return (tmp_11,)