import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = 0.15811388300841897 * in_0
        tmp_1 = tmp_0 + in_1
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        return (tmp_2,)