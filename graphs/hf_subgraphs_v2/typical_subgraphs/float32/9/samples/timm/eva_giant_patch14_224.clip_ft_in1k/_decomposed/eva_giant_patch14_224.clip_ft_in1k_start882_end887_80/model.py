import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.gelu(in_1, approximate='none')
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        tmp_4 = torch.nn.functional.linear(tmp_3, tmp_1, tmp_0)
        tmp_3 = tmp_1 = tmp_0 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False)
        tmp_4 = None
        tmp_6 = in_0 + tmp_5
        tmp_5 = None
        return (tmp_6,)