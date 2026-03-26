import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.max_pool2d(in_4, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
        tmp_5 = torch.nn.functional.batch_norm(tmp_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
        tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
        tmp_5 = None
        return (tmp_4, tmp_6)