import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (3, 3), (1, 1), 1);  in_3 = in_1 = in_0 = None
        tmp_4 = torch.flatten(conv2d, 2);  conv2d = None
        tmp_5 = torch.functional.norm(tmp_4, dim = -1, keepdim = True)
        tmp_6 = tmp_5 * 0.14433756729740643;  tmp_5 = None
        tmp_7 = tmp_6.clamp(min = 1e-05);  tmp_6 = None
        tmp_8 = tmp_4 / tmp_7;  tmp_4 = tmp_7 = None
        tmp_9 = tmp_8 * in_2;  tmp_8 = in_2 = None
        return (tmp_9,)
        