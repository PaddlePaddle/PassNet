import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5[slice(None, None, None), 0]
        tmp_6 = torch.nn.functional.linear(tmp_5, tmp_1, tmp_0)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = torch.tanh(tmp_6)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(in_6, tmp_3, None)
        tmp_3 = None
        tmp_9 = torch.nn.functional.linear(tmp_7, tmp_2, None)
        tmp_2 = None
        tmp_10 = tmp_8.norm(dim=-1, keepdim=True)
        tmp_11 = tmp_8 / tmp_10
        tmp_8 = tmp_10 = None
        tmp_12 = tmp_9.norm(dim=-1, keepdim=True)
        tmp_13 = tmp_9 / tmp_12
        tmp_9 = tmp_12 = None
        tmp_14 = tmp_4.exp()
        tmp_4 = None
        tmp_15 = tmp_11.t()
        tmp_16 = torch.matmul(tmp_13, tmp_15)
        tmp_15 = None
        tmp_17 = tmp_16 * tmp_14
        tmp_16 = tmp_14 = None
        tmp_18 = tmp_17.T
        return (tmp_7, tmp_11, tmp_13, tmp_17, tmp_18)