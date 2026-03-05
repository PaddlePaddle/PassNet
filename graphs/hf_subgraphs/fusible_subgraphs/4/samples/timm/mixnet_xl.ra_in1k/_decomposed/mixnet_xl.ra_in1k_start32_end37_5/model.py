import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.cat([in_4, in_5], 1)
        tmp_5 = torch.nn.functional.batch_norm(tmp_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_4 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_6 = torch.functional.split(tmp_5, [24, 24], 1)
        tmp_7 = tmp_6[0]
        tmp_8 = tmp_6[1]
        tmp_6 = None
        return (tmp_7, tmp_8, tmp_5)