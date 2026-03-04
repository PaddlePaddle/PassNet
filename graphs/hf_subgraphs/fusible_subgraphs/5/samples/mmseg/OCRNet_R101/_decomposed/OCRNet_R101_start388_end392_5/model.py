import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = 0.0625 * in_0
        tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
        tmp_0 = None
        tmp_2 = torch.matmul(tmp_1, in_1)
        tmp_1 = None
        tmp_3 = tmp_2.permute(0, 2, 1)
        tmp_2 = None
        return (tmp_3,)