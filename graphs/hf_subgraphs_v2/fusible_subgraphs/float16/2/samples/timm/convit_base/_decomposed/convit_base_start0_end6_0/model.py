import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        conv2d = torch.conv2d(in_4, in_1, in_0, (16, 16), (0, 0), (1, 1), 1);  in_4 = in_1 = in_0 = None
        tmp_6 = conv2d.flatten(2);  conv2d = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        tmp_8 = tmp_7 + in_3;  tmp_7 = in_3 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        tmp_10 = in_2.expand(1, -1, -1);  in_2 = None
        return (tmp_10, tmp_9)
        