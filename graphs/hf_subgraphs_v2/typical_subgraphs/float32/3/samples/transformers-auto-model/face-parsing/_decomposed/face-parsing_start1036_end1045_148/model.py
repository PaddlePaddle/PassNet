import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0.transpose(-1, -2)
        tmp_1 = torch.matmul(in_1, tmp_0)
        tmp_0 = None
        tmp_2 = tmp_1 / 8.0
        tmp_1 = None
        tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
        tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
        tmp_3 = None
        tmp_5 = torch.matmul(tmp_4, in_2)
        tmp_4 = None
        tmp_6 = tmp_5.permute(0, 2, 1, 3)
        tmp_5 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        tmp_8 = tmp_7.view((8, 1024, 320))
        tmp_7 = None
        return (tmp_8,)