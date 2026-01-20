import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3):
        tmp_0 = torch.cat([in_0, in_1], 1)
        tmp_1 = torch.nn.functional.batch_norm(tmp_0, w_2, w_3, w_1, w_0, False, 0.1, 1e-05)
        tmp_0 = None
        tmp_2 = torch.functional.split(tmp_1, [16, 16], 1)
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_2 = None
        return (tmp_1, tmp_3, tmp_4)