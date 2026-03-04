import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.embedding(tmp_1, tmp_3, 1, None, 2.0, False, False)
        tmp_1 = tmp_3 = None
        tmp_5 = torch.nn.functional.embedding(in_4, tmp_2, 1, None, 2.0, False, False)
        tmp_2 = None
        tmp_6 = tmp_4 + tmp_5
        tmp_4 = tmp_5 = None
        tmp_7 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_8 = tmp_6 * tmp_7
        tmp_6 = tmp_7 = None
        tmp_9 = tmp_8.to(torch.float32)
        tmp_8 = None
        return (tmp_9,)