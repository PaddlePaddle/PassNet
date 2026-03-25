import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.reshape(1, 197, 3, 16, 48);  in_0 = None
        tmp_1 = tmp_0.permute(2, 0, 3, 1, 4);  tmp_0 = None
        unbind = tmp_1.unbind(0);  tmp_1 = None
        tmp_3 = unbind[0]
        tmp_4 = unbind[1]
        tmp_5 = unbind[2];  unbind = None
        tmp_6 = tmp_4.transpose(-2, -1);  tmp_4 = None
        matmul = tmp_3 @ tmp_6;  tmp_3 = tmp_6 = None
        tmp_8 = matmul * 0.14433756729740643;  matmul = None
        tmp_9 = tmp_8.softmax(dim = -1);  tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False);  tmp_9 = None
        matmul_1 = tmp_10 @ tmp_5;  tmp_10 = tmp_5 = None
        tmp_12 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_13 = tmp_12.reshape(1, 197, 768);  tmp_12 = None
        return (tmp_13,)
        