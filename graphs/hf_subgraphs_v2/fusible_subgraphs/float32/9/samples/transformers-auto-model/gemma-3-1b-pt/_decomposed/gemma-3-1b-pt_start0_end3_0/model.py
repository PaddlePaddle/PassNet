import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.embedding(tmp_0, tmp_2, 0, None, 2.0, False, False)
        tmp_0 = tmp_2 = None
        tmp_4 = tmp_1.to(torch.float16)
        tmp_1 = None
        tmp_5 = tmp_3 * tmp_4
        tmp_3 = tmp_4 = None
        return (tmp_5,)