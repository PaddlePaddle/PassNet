import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = in_0[slice(None, None, None), 0, slice(None, None, None)]
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False)
        tmp_4 = None
        tmp_6 = torch.nn.functional.linear(tmp_5, tmp_1, tmp_0)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = torch.tanh(tmp_6)
        tmp_6 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.1, False, False)
        tmp_7 = None
        tmp_9 = torch.nn.functional.linear(tmp_8, tmp_3, tmp_2)
        tmp_8 = tmp_3 = tmp_2 = None
        return (tmp_9,)