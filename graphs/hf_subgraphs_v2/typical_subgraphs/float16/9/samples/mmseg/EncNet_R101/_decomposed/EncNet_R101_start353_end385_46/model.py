import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1):
        tmp_12 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_13 = tmp_12.view(1, 512, -1);  tmp_12 = None
        tmp_14 = tmp_13.transpose(1, 2);  tmp_13 = None
        tmp_15 = tmp_14.contiguous();  tmp_14 = None
        tmp_16 = w_3.view((1, 1, 32));  w_3 = None
        tmp_17 = tmp_15.unsqueeze(2)
        tmp_18 = tmp_17.expand((1, 4096, 32, 512));  tmp_17 = None
        tmp_19 = w_2.view((1, 1, 32, 512))
        tmp_20 = tmp_18 - tmp_19;  tmp_18 = tmp_19 = None
        tmp_21 = tmp_20.pow(2);  tmp_20 = None
        tmp_22 = tmp_21.sum(dim = 3);  tmp_21 = None
        tmp_23 = tmp_16 * tmp_22;  tmp_16 = tmp_22 = None
        tmp_24 = torch.nn.functional.softmax(tmp_23, dim = 2);  tmp_23 = None
        tmp_25 = w_2.view((1, 1, 32, 512));  w_2 = None
        tmp_26 = tmp_15.unsqueeze(2);  tmp_15 = None
        tmp_27 = tmp_26.expand((1, 4096, 32, 512));  tmp_26 = None
        tmp_28 = tmp_24.unsqueeze(3);  tmp_24 = None
        tmp_29 = tmp_27 - tmp_25;  tmp_27 = tmp_25 = None
        tmp_30 = tmp_28 * tmp_29;  tmp_28 = tmp_29 = None
        tmp_31 = tmp_30.sum(dim = 1);  tmp_30 = None
        tmp_32 = torch.nn.functional.batch_norm(tmp_31, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  tmp_31 = w_4 = w_5 = w_7 = w_6 = None
        tmp_33 = torch.nn.functional.relu(tmp_32, inplace = True);  tmp_32 = None
        tmp_34 = tmp_33.mean(dim = 1);  tmp_33 = None
        linear = torch.nn.functional.linear(tmp_34, w_9, w_8);  w_9 = w_8 = None
        tmp_36 = torch.sigmoid(linear);  linear = None
        tmp_37 = tmp_36.view(1, 512, 1, 1);  tmp_36 = None
        tmp_38 = in_0 * tmp_37;  tmp_37 = None
        tmp_39 = in_0 + tmp_38;  in_0 = tmp_38 = None
        tmp_40 = torch.relu_(tmp_39);  tmp_39 = None
        tmp_41 = torch.nn.functional.dropout2d(tmp_40, 0.1, False, False);  tmp_40 = None
        conv2d = torch.conv2d(tmp_41, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_41 = w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(tmp_34, w_11, w_10);  tmp_34 = w_11 = w_10 = None
        return (conv2d, linear_1)
        