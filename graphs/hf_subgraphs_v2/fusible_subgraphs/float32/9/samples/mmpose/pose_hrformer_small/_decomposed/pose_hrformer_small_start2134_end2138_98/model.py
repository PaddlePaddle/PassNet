import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.pad(in_0, (0, 0, 0, 1, 3, 3), 'constant', None)
        tmp_1 = tmp_0.view(1, 2, 7, 1, 7, 256)
        tmp_0 = None
        tmp_2 = tmp_1.permute(0, 1, 3, 2, 4, 5)
        tmp_1 = None
        tmp_3 = tmp_2.reshape(-1, 49, 256)
        tmp_2 = None
        return (tmp_3,)