import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
        tmp_5 = torch.nn.functional.interpolate(tmp_4, (128, 128), None, 'bilinear', False)
        tmp_4 = None
        tmp_6 = torch.cat([in_4, tmp_5], 1)
        tmp_5 = None
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
        tmp_6 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
        tmp_7 = None
        return (tmp_8,)