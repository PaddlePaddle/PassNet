import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_1 = tmp_0.view(1, 512, 64, 64)
        tmp_0 = None
        tmp_2 = in_0.view(512, 1, 1, 1)
        tmp_3 = torch.nn.functional.pad(tmp_1, (0, 0, 0, 0), 'constant', 0)
        tmp_1 = None
        return (tmp_2, tmp_3)