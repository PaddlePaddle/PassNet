import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor):
        tmp_30 = torch.nn.functional.batch_norm(in_30, in_24, in_25, in_27, in_26, False, 0.1, 1e-05);  in_24 = in_25 = in_27 = in_26 = None
        conv2d = torch.conv2d(tmp_30, in_11, in_10, (1, 1), (0, 0), (1, 1), 1);  in_11 = in_10 = None
        tmp_32 = torch.nn.functional.gelu(conv2d);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_32, in_5, in_4, (1, 1), (2, 2), (1, 1), 128);  in_5 = in_4 = None
        conv2d_2 = torch.conv2d(conv2d_1, in_3, in_2, (1, 1), (9, 9), (3, 3), 128);  conv2d_1 = in_3 = in_2 = None
        conv2d_3 = torch.conv2d(conv2d_2, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  conv2d_2 = in_7 = in_6 = None
        tmp_36 = tmp_32 * conv2d_3;  tmp_32 = conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_36, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  tmp_36 = in_9 = in_8 = None
        tmp_38 = conv2d_4 + tmp_30;  conv2d_4 = tmp_30 = None
        tmp_39 = in_12.unsqueeze(-1);  in_12 = None
        tmp_40 = tmp_39.unsqueeze(-1);  tmp_39 = None
        tmp_41 = tmp_40 * tmp_38;  tmp_40 = tmp_38 = None
        tmp_42 = in_30 + tmp_41;  in_30 = tmp_41 = None
        tmp_43 = torch.nn.functional.batch_norm(tmp_42, in_20, in_21, in_23, in_22, False, 0.1, 1e-05);  in_20 = in_21 = in_23 = in_22 = None
        conv2d_5 = torch.conv2d(tmp_43, in_16, in_15, (1, 1), (0, 0), (1, 1), 1);  tmp_43 = in_16 = in_15 = None
        conv2d_6 = torch.conv2d(conv2d_5, in_14, in_13, (1, 1), (1, 1), (1, 1), 512);  conv2d_5 = in_14 = in_13 = None
        tmp_46 = torch.nn.functional.gelu(conv2d_6);  conv2d_6 = None
        tmp_47 = torch.nn.functional.dropout(tmp_46, 0.0, False, False);  tmp_46 = None
        conv2d_7 = torch.conv2d(tmp_47, in_18, in_17, (1, 1), (0, 0), (1, 1), 1);  tmp_47 = in_18 = in_17 = None
        tmp_49 = torch.nn.functional.dropout(conv2d_7, 0.0, False, False);  conv2d_7 = None
        tmp_50 = in_19.unsqueeze(-1);  in_19 = None
        tmp_51 = tmp_50.unsqueeze(-1);  tmp_50 = None
        tmp_52 = tmp_51 * tmp_49;  tmp_51 = tmp_49 = None
        tmp_53 = tmp_42 + tmp_52;  tmp_42 = tmp_52 = None
        tmp_54 = tmp_53.flatten(2);  tmp_53 = None
        tmp_55 = tmp_54.transpose(1, 2);  tmp_54 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (128,), in_29, in_28, 1e-06);  tmp_55 = in_29 = in_28 = None
        tmp_57 = tmp_56.view(512, 7, 7, 128);  tmp_56 = None
        tmp_58 = tmp_57.permute(0, 3, 1, 2);  tmp_57 = None
        tmp_59 = tmp_58.mean(dim = [-2, -1]);  tmp_58 = None
        linear = torch.nn.functional.linear(tmp_59, in_1, in_0);  tmp_59 = in_1 = in_0 = None
        return (linear,)
        