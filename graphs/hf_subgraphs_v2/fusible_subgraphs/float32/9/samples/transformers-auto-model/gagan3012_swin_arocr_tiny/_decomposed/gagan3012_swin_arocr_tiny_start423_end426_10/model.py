import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.clamp(tmp_0, max=4.605170185988092)
        tmp_0 = None
        tmp_2 = tmp_1.exp()
        tmp_1 = None
        tmp_3 = in_1 * tmp_2
        tmp_2 = None
        return (tmp_3,)