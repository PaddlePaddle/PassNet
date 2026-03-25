import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear.view((1, 64, -1, 64));  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_4.transpose(2, 3);  in_4 = None
        matmul = torch.matmul(in_5, tmp_5);  in_5 = tmp_5 = None
        tmp_7 = matmul * 0.125;  matmul = None
        tmp_8 = in_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))];  in_3 = None
        tmp_9 = tmp_7 + tmp_8;  tmp_7 = tmp_8 = None
        tmp_10 = torch.nn.functional.softmax(tmp_9, dim = -1, dtype = torch.float32);  tmp_9 = None
        tmp_11 = tmp_10.to(torch.float32);  tmp_10 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, p = 0.0, training = False);  tmp_11 = None
        to_1 = tmp_12.to(torch.bfloat16);  tmp_12 = None
        matmul_1 = torch.matmul(to_1, tmp_4);  to_1 = tmp_4 = None
        tmp_14 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_15 = tmp_14.contiguous();  tmp_14 = None
        tmp_16 = tmp_15.reshape(1, 64, -1);  tmp_15 = None
        tmp_17 = tmp_16.contiguous();  tmp_16 = None
        return (tmp_17,)
        