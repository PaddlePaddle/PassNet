import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = tmp_0[in_3]
        tmp_0 = None
        tmp_2 = tmp_1.view(49, 49, -1)
        tmp_1 = None
        tmp_3 = tmp_2.permute(2, 0, 1)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = tmp_4.unsqueeze(0)
        tmp_4 = None
        tmp_6 = in_1 + tmp_5
        tmp_5 = None
        tmp_7 = tmp_6.view(1, 64, 6, 49, 49)
        tmp_6 = None
        tmp_8 = in_2.unsqueeze(1)
        tmp_9 = tmp_8.unsqueeze(0)
        tmp_8 = None
        tmp_10 = tmp_7 + tmp_9
        tmp_7 = tmp_9 = None
        tmp_11 = tmp_10.view(-1, 6, 49, 49)
        tmp_10 = None
        tmp_12 = torch.nn.functional.softmax(tmp_11, dim=-1)
        tmp_11 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
        tmp_12 = None
        return (tmp_13,)