import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
        tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
        tmp_2 = None
        tmp_4 = tmp_3.mul(tmp_0)
        tmp_0 = None
        tmp_5 = tmp_4.reshape(16, 17, -1)
        tmp_4 = None
        tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
        tmp_5 = None
        tmp_7 = tmp_3.mul(tmp_1)
        tmp_1 = None
        tmp_8 = tmp_7.reshape(16, 17, -1)
        tmp_7 = None
        tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
        tmp_8 = None
        tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
        tmp_6 = tmp_9 = None
        return (tmp_3, tmp_10)