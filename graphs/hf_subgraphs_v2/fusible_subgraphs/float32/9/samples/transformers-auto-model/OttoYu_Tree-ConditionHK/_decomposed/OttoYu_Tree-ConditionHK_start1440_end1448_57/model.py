import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        tmp_4 = tmp_3.view(-1, 7, 7, 1024)
        tmp_3 = None
        tmp_5 = tmp_4.view(-1, 1, 1, 7, 7, 1024)
        tmp_4 = None
        tmp_6 = tmp_5.permute(0, 1, 3, 2, 4, 5)
        tmp_5 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        tmp_8 = tmp_7.view(-1, 7, 7, 1024)
        tmp_7 = None
        tmp_9 = tmp_8.view(1, 49, 1024)
        tmp_8 = None
        return (tmp_9,)