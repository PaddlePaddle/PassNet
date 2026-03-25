import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = torch.nn.functional.relu(in_7, inplace = True);  in_7 = None
        to = tmp_6.to(torch.bfloat16);  tmp_6 = None
        conv2d = torch.conv2d(to, in_0, None, (1, 1), (0, 0), (1, 1), 1);  to = in_0 = None
        tmp_8 = conv2d + in_6;  conv2d = in_6 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  tmp_8 = in_2 = in_3 = in_5 = in_4 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace = True);  tmp_9 = None
        to_1 = tmp_10.to(torch.bfloat16)
        conv2d_1 = torch.conv2d(to_1, in_1, None, (2, 2), (0, 0), (1, 1), 1);  to_1 = in_1 = None
        return (conv2d_1, tmp_10)
        