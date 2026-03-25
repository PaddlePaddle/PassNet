import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = torch.nn.functional.relu(in_3, inplace = True);  in_3 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, [128, 128], None, 'bilinear', False);  tmp_2 = None
        tmp_4 = torch.nn.functional.interpolate(tmp_3, (128, 128), None, 'bilinear', False);  tmp_3 = None
        tmp_5 = in_2 + tmp_4;  in_2 = tmp_4 = None
        tmp_6 = torch.nn.functional.dropout2d(tmp_5, 0.1, False, False);  tmp_5 = None
        conv2d = torch.conv2d(tmp_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = in_1 = in_0 = None
        return (conv2d,)
        