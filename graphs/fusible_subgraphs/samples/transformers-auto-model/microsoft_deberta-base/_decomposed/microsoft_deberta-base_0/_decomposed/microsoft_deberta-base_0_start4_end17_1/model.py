import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_1 + in_0
        tmp_1 = tmp_0.float()
        tmp_0 = None
        tmp_2 = tmp_1.mean(-1, keepdim=True)
        tmp_3 = tmp_1 - tmp_2
        tmp_4 = tmp_3.pow(2)
        tmp_3 = None
        tmp_5 = tmp_4.mean(-1, keepdim=True)
        tmp_4 = None
        tmp_6 = tmp_1 - tmp_2
        tmp_1 = tmp_2 = None
        tmp_7 = tmp_5 + 1e-07
        tmp_5 = None
        tmp_8 = torch.sqrt(tmp_7)
        tmp_7 = None
        tmp_9 = tmp_6 / tmp_8
        tmp_6 = tmp_8 = None
        tmp_10 = tmp_9.to(torch.float32)
        tmp_9 = None
        tmp_11 = w_1 * tmp_10
        tmp_10 = None
        tmp_12 = tmp_11 + w_0
        tmp_11 = None
        return (tmp_12,)