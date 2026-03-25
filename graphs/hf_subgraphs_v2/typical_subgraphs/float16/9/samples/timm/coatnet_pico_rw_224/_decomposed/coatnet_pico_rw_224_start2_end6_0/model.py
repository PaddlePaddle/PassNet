import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, in_0 : torch.Tensor):
        tmp_5 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_5, w_4, None, (1, 1), (1, 1), (1, 1), 1);  tmp_5 = w_4 = None
        tmp_7 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
        tmp_8 = torch.nn.functional.batch_norm(conv2d, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  conv2d = w_0 = w_1 = w_3 = w_2 = None
        return (tmp_7, tmp_8)
        