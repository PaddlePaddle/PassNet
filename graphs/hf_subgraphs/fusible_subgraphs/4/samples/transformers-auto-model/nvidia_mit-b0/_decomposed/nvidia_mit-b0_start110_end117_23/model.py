import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 / 5.656854249492381
        tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
        tmp_1 = None
        tmp_3 = torch.matmul(tmp_2, in_1)
        tmp_2 = None
        tmp_4 = tmp_3.permute(0, 2, 1, 3)
        tmp_3 = None
        tmp_5 = tmp_4.contiguous()
        tmp_4 = None
        tmp_6 = tmp_5.view((12, 4096, 64))
        tmp_5 = None
        return (tmp_6,)