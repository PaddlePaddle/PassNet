import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 128);  in_2 = in_1 = in_0 = None
        tmp_3 = conv2d.flatten(2);  conv2d = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = torch.nn.functional.gelu(tmp_4);  tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False);  tmp_5 = None
        return (tmp_6,)
        