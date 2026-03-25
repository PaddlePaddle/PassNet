import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2 + in_3
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.view(1, 7, 7, 768)
        tmp_3 = None
        tmp_5 = torch.nn.functional.pad(tmp_4, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_4 = None
        tmp_6 = tmp_5.view(1, 1, 7, 1, 7, 768)
        tmp_5 = None
        tmp_7 = tmp_6.permute(0, 1, 3, 2, 4, 5)
        tmp_6 = None
        tmp_8 = tmp_7.contiguous()
        tmp_7 = None
        tmp_9 = tmp_8.view(-1, 7, 7, 768)
        tmp_8 = None
        tmp_10 = tmp_9.view(-1, 49, 768)
        tmp_9 = None
        return (tmp_10, tmp_2)