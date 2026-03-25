import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        conv2d = torch.conv2d(in_4, in_2, in_1, (14, 14), 'valid', (1, 1), 1);  in_4 = in_2 = in_1 = None
        tmp_5 = conv2d.flatten(2);  conv2d = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = torch.nn.functional.embedding(in_0, in_3, None, None, 2.0, False, False);  in_0 = in_3 = None
        tmp_8 = tmp_6 + tmp_7;  tmp_6 = tmp_7 = None
        return (tmp_8,)
        