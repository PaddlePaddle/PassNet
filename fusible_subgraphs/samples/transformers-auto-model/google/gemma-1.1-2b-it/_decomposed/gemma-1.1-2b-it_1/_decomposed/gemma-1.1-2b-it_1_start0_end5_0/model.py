import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0):
        tmp_0 = torch.nn.functional.linear(in_1, w_0, None)
        tmp_1 = tmp_0.view((1, 3, -1, 256))
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = in_0.unsqueeze(1)
        tmp_4 = in_2.unsqueeze(1)
        return (tmp_3, tmp_4, tmp_2)