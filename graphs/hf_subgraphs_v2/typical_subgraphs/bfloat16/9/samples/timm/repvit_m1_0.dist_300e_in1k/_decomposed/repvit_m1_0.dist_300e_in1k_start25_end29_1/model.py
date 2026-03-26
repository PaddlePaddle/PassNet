import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_5, w_4, (1, 1), (0, 0), (1, 1), 56);  w_5 = w_4 = None
        tmp_7 = in_1 + conv2d;  in_1 = conv2d = None
        tmp_8 = tmp_7 + in_0;  tmp_7 = in_0 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_8 = w_0 = w_1 = w_3 = w_2 = None
        return (tmp_9,)
        