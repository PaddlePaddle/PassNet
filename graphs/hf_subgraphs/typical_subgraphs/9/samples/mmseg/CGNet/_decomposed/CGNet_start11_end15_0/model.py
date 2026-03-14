import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = torch.prelu(in_1, tmp_5)
        tmp_5 = None
        tmp_7 = torch.cat([tmp_6, in_0], 1)
        tmp_6 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
        tmp_7 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_9 = torch.prelu(tmp_8, tmp_4)
        tmp_8 = tmp_4 = None
        return (tmp_9,)