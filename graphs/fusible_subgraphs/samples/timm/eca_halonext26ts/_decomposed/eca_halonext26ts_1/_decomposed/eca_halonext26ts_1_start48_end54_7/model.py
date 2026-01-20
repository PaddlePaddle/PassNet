import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3):
        tmp_0 = in_0.reshape(-1, 4, 4, 2, 2)
        tmp_1 = tmp_0.permute(0, 3, 1, 4, 2)
        tmp_0 = None
        tmp_2 = tmp_1.contiguous()
        tmp_1 = None
        tmp_3 = tmp_2.view(1, 512, 8, 8)
        tmp_2 = None
        tmp_4 = torch.nn.functional.batch_norm(tmp_3, w_0, w_1, w_3, w_2, False, 0.1, 1e-05)
        tmp_3 = None
        tmp_5 = torch.nn.functional.silu(tmp_4, inplace=True)
        tmp_4 = None
        return (tmp_5,)