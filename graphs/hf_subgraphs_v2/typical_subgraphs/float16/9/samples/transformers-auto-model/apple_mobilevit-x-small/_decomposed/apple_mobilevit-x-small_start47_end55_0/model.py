import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, in_0 : torch.Tensor):
        tmp_3 = torch.nn.functional.silu(in_0, inplace = False);  in_0 = None
        conv2d = torch.conv2d(tmp_3, w_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_3 = w_0 = None
        tmp_5 = conv2d.reshape(1536, 2, 16, 2);  conv2d = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = tmp_6.reshape(1, 96, 256, 4);  tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 3);  tmp_7 = None
        tmp_9 = tmp_8.reshape(4, 256, -1);  tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (96,), w_2, w_1, 1e-05);  w_2 = w_1 = None
        return (tmp_10, tmp_9)
        