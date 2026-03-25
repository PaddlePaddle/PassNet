import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_2.view(4, -1, 1024)
        tmp_2 = torch.nn.functional.linear(tmp_1, tmp_0, None)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
        tmp_2 = None
        tmp_4 = in_3 + tmp_3
        tmp_3 = None
        tmp_5 = in_1[-1]
        tmp_6 = tmp_5 + 1
        tmp_5 = tmp_6 = None
        tmp_7 = tmp_4.to(torch.float32)
        tmp_8 = tmp_7.pow(2)
        tmp_7 = None
        tmp_9 = tmp_8.mean(-1, keepdim=True)
        tmp_8 = None
        tmp_10 = tmp_9 + 1e-06
        tmp_9 = None
        tmp_11 = torch.rsqrt(tmp_10)
        tmp_10 = None
        return (tmp_4, tmp_11)