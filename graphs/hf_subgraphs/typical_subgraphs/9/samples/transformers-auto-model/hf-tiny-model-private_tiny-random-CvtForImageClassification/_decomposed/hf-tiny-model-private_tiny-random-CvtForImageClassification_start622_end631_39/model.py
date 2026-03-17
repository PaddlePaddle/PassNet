import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.gelu(in_1, approximate='none')
        tmp_3 = torch.nn.functional.linear(tmp_2, tmp_1, tmp_0)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
        tmp_3 = None
        tmp_5 = tmp_4 + in_0
        tmp_4 = None
        tmp_6 = torch.functional.split(tmp_5, [1, 16], 1)
        tmp_5 = None
        tmp_7 = tmp_6[0]
        tmp_8 = tmp_6[1]
        tmp_6 = None
        tmp_9 = tmp_8.permute(0, 2, 1)
        tmp_8 = None
        tmp_10 = tmp_9.view(1, 96, 4, 4)
        tmp_9 = None
        return (tmp_7, tmp_10)