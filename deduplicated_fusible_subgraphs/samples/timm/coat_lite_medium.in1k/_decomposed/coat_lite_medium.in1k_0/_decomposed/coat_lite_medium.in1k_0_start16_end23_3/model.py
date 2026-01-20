import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = tmp_0.reshape(1, 3137, 3, 8, 16)
        tmp_0 = None
        tmp_2 = tmp_1.permute(2, 0, 3, 1, 4)
        tmp_1 = None
        tmp_3 = tmp_2.unbind(0)
        tmp_2 = None
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_6 = tmp_3[2]
        tmp_3 = None
        return (tmp_4, tmp_5, tmp_6)