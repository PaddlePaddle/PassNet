import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.layer_norm(in_2, (192,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view(1, 28, 28, 192)
        tmp_2 = None
        tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 4, 7, 4, 7, 192)
        tmp_4 = None
        tmp_6 = tmp_5.permute(0, 1, 3, 2, 4, 5)
        tmp_5 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        tmp_8 = tmp_7.view(-1, 7, 7, 192)
        tmp_7 = None
        tmp_9 = tmp_8.view(-1, 49, 192)
        tmp_8 = None
        return (tmp_9,)