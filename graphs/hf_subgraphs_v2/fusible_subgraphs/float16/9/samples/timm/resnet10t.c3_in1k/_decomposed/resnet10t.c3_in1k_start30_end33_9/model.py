import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1);  in_6 = in_4 = None
        tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d = in_0 = in_1 = in_3 = in_2 = None
        tmp_7 = torch.nn.functional.avg_pool2d(in_5, 2, 2, 0, True, False, None);  in_5 = None
        return (tmp_7, tmp_6)
        