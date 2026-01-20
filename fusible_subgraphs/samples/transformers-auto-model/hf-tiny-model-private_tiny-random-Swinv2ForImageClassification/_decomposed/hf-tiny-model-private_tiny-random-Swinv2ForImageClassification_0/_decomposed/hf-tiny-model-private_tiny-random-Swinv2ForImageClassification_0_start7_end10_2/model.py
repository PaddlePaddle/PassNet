import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = torch.clamp(w_0, max=4.605170185988092)
        tmp_1 = tmp_0.exp()
        tmp_0 = None
        tmp_2 = in_0 * tmp_1
        tmp_1 = None
        return (tmp_2,)