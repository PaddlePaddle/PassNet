import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.embedding(tmp_3, tmp_2, None, None, 2.0, False, False)
        tmp_3 = tmp_2 = None
        tmp_5 = in_4 + in_5
        tmp_6 = tmp_5 + tmp_4
        tmp_5 = tmp_4 = None
        tmp_7 = tmp_6 * tmp_1
        tmp_6 = tmp_1 = None
        tmp_8 = tmp_7 + tmp_0
        tmp_7 = tmp_0 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        return (tmp_9,)