import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_0, inplace=False)
        tmp_3 = tmp_1 * tmp_2
        tmp_1 = tmp_2 = None
        tmp_4 = tmp_3 + tmp_0
        tmp_3 = tmp_0 = None
        tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
        tmp_4 = None
        return (tmp_5,)