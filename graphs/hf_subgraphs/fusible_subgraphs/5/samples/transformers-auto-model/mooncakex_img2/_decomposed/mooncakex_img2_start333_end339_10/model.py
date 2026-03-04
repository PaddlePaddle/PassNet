import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.softmax(in_0, -1, _stacklevel=5)
        tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
        tmp_0 = None
        tmp_2 = torch.matmul(tmp_1, in_1)
        tmp_1 = None
        tmp_3 = tmp_2.permute(0, 2, 1, 3)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = tmp_4.view(2, 7, 768)
        tmp_4 = None
        return (tmp_5,)