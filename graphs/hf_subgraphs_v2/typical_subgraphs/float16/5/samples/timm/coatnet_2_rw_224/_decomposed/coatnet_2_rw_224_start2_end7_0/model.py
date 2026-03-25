import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_5 = torch.nn.functional.silu(in_5, inplace = True);  in_5 = None
        to = tmp_5.to(torch.float16);  tmp_5 = None
        conv2d = torch.conv2d(to, in_4, None, (1, 1), (1, 1), (1, 1), 1);  to = in_4 = None
        tmp_7 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
        tmp_8 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d = in_0 = in_1 = in_3 = in_2 = None
        tmp_9 = torch.nn.functional.silu(tmp_8, inplace = True);  tmp_8 = None
        return (tmp_7, tmp_9)
        