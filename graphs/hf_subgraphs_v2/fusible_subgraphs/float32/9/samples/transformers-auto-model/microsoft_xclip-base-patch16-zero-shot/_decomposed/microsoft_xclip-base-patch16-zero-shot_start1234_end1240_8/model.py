import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
        tmp_2 = in_1 / tmp_1
        tmp_1 = None
        tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
        tmp_4 = in_2 / tmp_3
        tmp_3 = None
        tmp_5 = tmp_0.exp()
        tmp_0 = None
        tmp_6 = tmp_5 * tmp_4
        tmp_5 = None
        return (tmp_6, tmp_4, tmp_2)