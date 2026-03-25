import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_1 = in_3.mean((2, 3))
        tmp_2 = in_1 // 8;  in_1 = None
        tmp_3 = torch.sym_sum([1, tmp_2]);  tmp_2 = tmp_3 = None
        tmp_4 = tmp_1.view(1, 1, -1);  tmp_1 = None
        conv1d = torch.conv1d(tmp_4, in_0, None, (1,), (2,), (1,), 1);  tmp_4 = in_0 = None
        tmp_6 = conv1d.sigmoid();  conv1d = None
        tmp_7 = tmp_6.view(1, -1, 1, 1);  tmp_6 = None
        tmp_8 = tmp_7.expand_as(in_3);  tmp_7 = None
        tmp_9 = in_3 * tmp_8;  in_3 = tmp_8 = None
        tmp_10 = torch.nn.functional.avg_pool2d(in_2, 2, 2, 0, True, False, None);  in_2 = None
        return (tmp_10, tmp_9)
        