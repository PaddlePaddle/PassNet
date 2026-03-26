import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = in_0[slice(None, None, None), 0, slice(None, None, None)]
        tmp_3 = torch.nn.functional.linear(tmp_2, tmp_0, None)
        tmp_2 = tmp_0 = None
        tmp_4 = in_1.norm(p=2, dim=-1, keepdim=True)
        tmp_5 = in_1 / tmp_4
        tmp_4 = None
        tmp_6 = tmp_3.norm(p=2, dim=-1, keepdim=True)
        tmp_7 = tmp_3 / tmp_6
        tmp_3 = tmp_6 = None
        tmp_8 = tmp_1.exp()
        tmp_1 = None
        tmp_9 = tmp_5.t()
        tmp_10 = torch.matmul(tmp_7, tmp_9)
        tmp_9 = None
        tmp_11 = tmp_10 * tmp_8
        tmp_10 = tmp_8 = None
        tmp_12 = tmp_11.t()
        return (tmp_5, tmp_7, tmp_11, tmp_12)