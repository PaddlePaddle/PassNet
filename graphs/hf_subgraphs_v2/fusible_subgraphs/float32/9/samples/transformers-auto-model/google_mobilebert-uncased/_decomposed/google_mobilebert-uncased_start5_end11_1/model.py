import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.embedding(tmp_0, tmp_1, 0, None, 2.0, False, False)
        tmp_0 = tmp_1 = None
        tmp_3 = tmp_2[slice(None, None, None), slice(1, None, None)]
        tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
        tmp_3 = None
        tmp_5 = tmp_2[slice(None, None, None), slice(None, -1, None)]
        tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
        tmp_5 = None
        tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
        tmp_4 = tmp_2 = tmp_6 = None
        return (tmp_7,)