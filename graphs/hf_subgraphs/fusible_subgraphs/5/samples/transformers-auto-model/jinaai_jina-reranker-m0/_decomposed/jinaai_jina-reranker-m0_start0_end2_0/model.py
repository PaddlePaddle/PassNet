import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.embedding(tmp_1, tmp_2, None, None, 2.0, False, False)
        tmp_1 = tmp_2 = None
        tmp_4 = tmp_0.long()
        tmp_0 = None
        return (tmp_3, tmp_4)