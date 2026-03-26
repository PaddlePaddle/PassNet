import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3.contiguous()
        tmp_3 = tmp_2.view(-1, 35, 35, 384)
        tmp_2 = None
        tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
        tmp_3 = None
        tmp_5 = tmp_4[slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None)]
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = None
        tmp_7 = tmp_6.view(1, 1024, 384)
        tmp_6 = None
        tmp_8 = in_2 + tmp_7
        tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        return (tmp_8, tmp_9)