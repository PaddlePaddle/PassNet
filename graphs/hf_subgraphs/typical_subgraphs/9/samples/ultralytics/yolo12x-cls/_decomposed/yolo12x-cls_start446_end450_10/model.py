import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = torch.nn.functional.silu(in_0, inplace=False)
        tmp_2 = tmp_0.view(-1, 768, 1, 1)
        tmp_0 = None
        tmp_3 = tmp_2 * tmp_1
        tmp_2 = tmp_1 = None
        tmp_4 = in_1 + tmp_3
        tmp_3 = None
        return (tmp_4,)