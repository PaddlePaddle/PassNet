import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
        tmp_3 = tmp_1 * tmp_2
        tmp_1 = tmp_2 = None
        tmp_4 = tmp_3 + tmp_0
        tmp_3 = tmp_0 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        tmp_4 = None
        return (tmp_5,)