import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.batch_norm(in_0, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.01, 0.001)
        tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        return (tmp_4,)