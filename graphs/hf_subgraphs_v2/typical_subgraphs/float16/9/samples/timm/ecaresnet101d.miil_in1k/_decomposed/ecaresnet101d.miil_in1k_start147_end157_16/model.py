import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1, in_2):
        tmp_1 = in_2.mean((2, 3))
        tmp_2 = in_0 // 16;  in_0 = None
        tmp_3 = torch.sym_sum([1, tmp_2]);  tmp_2 = tmp_3 = None
        tmp_4 = tmp_1.view(1, 1, -1);  tmp_1 = None
        conv1d = torch.conv1d(tmp_4, w_0, None, (1,), (2,), (1,), 1);  tmp_4 = w_0 = None
        tmp_6 = conv1d.sigmoid();  conv1d = None
        tmp_7 = tmp_6.view(1, -1, 1, 1);  tmp_6 = None
        tmp_8 = tmp_7.expand_as(in_2);  tmp_7 = None
        tmp_9 = in_2 * tmp_8;  in_2 = tmp_8 = None
        tmp_10 = torch.nn.functional.avg_pool2d(in_1, 2, 2, 0, True, False, None);  in_1 = None
        return (tmp_10, tmp_9)
        