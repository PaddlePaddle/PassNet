import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_2, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_2 = w_1 = w_0 = None
        tmp_4 = conv2d.sigmoid();  conv2d = None
        tmp_5 = in_1 * tmp_4;  in_1 = tmp_4 = None
        tmp_6 = torch.nn.functional.hardtanh(tmp_5, 0.0, 6.0, False);  tmp_5 = None
        return (tmp_6,)
        