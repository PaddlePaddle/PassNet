import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.batch_norm(in_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_5 = torch.nn.functional.relu(tmp_4, inplace=False)
        tmp_4 = None
        return (tmp_5,)