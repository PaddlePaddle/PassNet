import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        matmul = in_1 @ in_3;  in_1 = in_3 = None
        tmp_1 = matmul.reshape(-1, 8, 23);  matmul = None
        tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None);  tmp_1 = None
        tmp_3 = tmp_2.flatten(1);  tmp_2 = None
        tmp_4 = torch.nn.functional.pad(tmp_3, [0, 15], 'constant', None);  tmp_3 = None
        tmp_5 = tmp_4.reshape(-1, 9, 23);  tmp_4 = None
        tmp_6 = tmp_5[(slice(None, None, None), slice(None, 8, None), slice(11, None, None))];  tmp_5 = None
        tmp_7 = tmp_6.reshape(8, 8, 1, 8, 12);  tmp_6 = None
        tmp_8 = tmp_7.expand(-1, -1, 12, -1, -1);  tmp_7 = None
        tmp_9 = tmp_8.permute((0, 3, 1, 4, 2));  tmp_8 = None
        tmp_10 = tmp_9 + in_2;  tmp_9 = in_2 = None
        tmp_11 = tmp_10.reshape(8, 1, 64, -1);  tmp_10 = None
        tmp_12 = in_0 + tmp_11;  in_0 = tmp_11 = None
        tmp_13 = tmp_12.softmax(dim = -1);  tmp_12 = None
        return (tmp_13,)
        