import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = in_0.transpose(1, 2);  in_0 = None
        tmp_3 = tmp_2.view(1, 2048, 16, 16);  tmp_2 = None
        conv2d = torch.conv2d(tmp_3, w_1, w_0, (1, 1), (1, 1), (1, 1), 2048);  tmp_3 = w_1 = w_0 = None
        tmp_5 = conv2d.flatten(2);  conv2d = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = torch.nn.functional.gelu(tmp_6);  tmp_6 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False);  tmp_7 = None
        return (tmp_8,)
        