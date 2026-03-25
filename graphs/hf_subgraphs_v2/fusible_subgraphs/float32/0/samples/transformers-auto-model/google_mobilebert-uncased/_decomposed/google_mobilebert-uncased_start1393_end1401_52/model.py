import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0 / 5.656854249492381
        tmp_1 = tmp_0 + in_1
        tmp_0 = None
        tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
        tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
        tmp_2 = None
        tmp_4 = torch.matmul(tmp_3, in_2)
        tmp_3 = None
        tmp_5 = tmp_4.permute(0, 2, 1, 3)
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = None
        tmp_7 = tmp_6.view((1, 64, 128))
        tmp_6 = None
        return (tmp_7,)