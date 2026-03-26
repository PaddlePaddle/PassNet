import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_0[slice(None, None, None), 0]
        tmp_5 = torch.nn.functional.linear(tmp_4, tmp_1, tmp_0)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = torch.tanh(tmp_5)
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.1, False, False)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_3, tmp_2)
        tmp_7 = tmp_3 = tmp_2 = None
        return (tmp_8,)