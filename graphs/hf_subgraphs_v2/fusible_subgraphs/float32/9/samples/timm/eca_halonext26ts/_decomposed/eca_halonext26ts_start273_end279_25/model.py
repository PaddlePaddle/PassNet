import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4.reshape(-1, 8, 8, 1, 1)
        tmp_5 = tmp_4.permute(0, 3, 1, 4, 2)
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = None
        tmp_7 = tmp_6.view(1, 512, 8, 8)
        tmp_6 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_7 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_9 = torch.nn.functional.silu(tmp_8, inplace=True)
        tmp_8 = None
        return (tmp_9,)