import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3.chunk(2, dim=-1)
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_2 = None
        tmp_5 = torch.nn.functional.silu(tmp_3, inplace=False)
        tmp_3 = None
        tmp_6 = tmp_5 * tmp_4
        tmp_5 = tmp_4 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_1, tmp_0)
        tmp_7 = tmp_1 = tmp_0 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        tmp_10 = in_2 + tmp_9
        tmp_9 = None
        return (tmp_10,)