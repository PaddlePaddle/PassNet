import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.softmax(in_1, dim=1)
        tmp_1 = in_0 * tmp_0
        tmp_0 = None
        tmp_2 = torch.sum(tmp_1, dim=1)
        tmp_1 = None
        return (tmp_2,)