import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor):
        tmp_32 = torch.nn.functional.relu(in_32, inplace = True);  in_32 = None
        tmp_33 = torch.nn.functional.dropout2d(tmp_32, 0.1, False, False)
        conv2d = torch.conv2d(tmp_33, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_33 = in_0 = None
        tmp_35 = in_1.clone();  in_1 = None
        tmp_36 = tmp_35[None];  tmp_35 = None
        tmp_37 = tmp_36.expand(1, 150, 512, 1, 1);  tmp_36 = None
        conv2d_1 = torch.conv2d(tmp_32, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  in_9 = in_8 = None
        tmp_39 = conv2d.softmax(dim = 1);  conv2d = None
        einsum = torch.functional.einsum('bnhw,bchw->bnc', tmp_39, conv2d_1);  tmp_39 = None
        tmp_41 = tmp_37.reshape(1, 150, 512, -1);  tmp_37 = None
        tmp_42 = tmp_41.permute(0, 1, 3, 2);  tmp_41 = None
        tmp_43 = einsum.reshape(-1, 256);  einsum = None
        linear = torch.nn.functional.linear(tmp_43, in_13, in_12);  tmp_43 = in_13 = in_12 = None
        tmp_45 = linear[(slice(None, None, None), slice(None, 256, None))]
        tmp_46 = tmp_45.view(-1, 256);  tmp_45 = None
        tmp_47 = linear[(slice(None, None, None), slice(-256, None, None))];  linear = None
        tmp_48 = tmp_47.view(-1, 256);  tmp_47 = None
        tmp_49 = tmp_42.reshape(300, -1, 256);  tmp_42 = None
        linear_1 = torch.nn.functional.linear(tmp_49, in_21, in_20);  tmp_49 = in_21 = in_20 = None
        tmp_51 = linear_1[(Ellipsis, slice(None, 256, None))]
        tmp_52 = linear_1[(Ellipsis, slice(-256, None, None))];  linear_1 = None
        tmp_53 = tmp_46.unsqueeze(-2);  tmp_46 = None
        tmp_54 = tmp_51 * tmp_53;  tmp_51 = tmp_53 = None
        linear_2 = torch.nn.functional.linear(tmp_54, in_19, in_18);  in_19 = in_18 = None
        tmp_56 = torch.nn.functional.layer_norm(linear_2, (256,), in_23, in_22, 1e-05);  linear_2 = in_23 = in_22 = None
        linear_3 = torch.nn.functional.linear(tmp_54, in_31, in_30);  tmp_54 = in_31 = in_30 = None
        tmp_58 = torch.nn.functional.layer_norm(linear_3, (256,), in_27, in_26, 1e-05);  linear_3 = in_27 = in_26 = None
        tmp_59 = tmp_56.sigmoid();  tmp_56 = None
        tmp_60 = tmp_58.sigmoid();  tmp_58 = None
        tmp_61 = torch.nn.functional.layer_norm(tmp_48, (256,), in_29, in_28, 1e-05);  tmp_48 = in_29 = in_28 = None
        tmp_62 = torch.nn.functional.layer_norm(tmp_52, (256,), in_25, in_24, 1e-05);  tmp_52 = in_25 = in_24 = None
        tmp_63 = tmp_61.unsqueeze(-2);  tmp_61 = None
        tmp_64 = tmp_60 * tmp_63;  tmp_60 = tmp_63 = None
        tmp_65 = tmp_59 * tmp_62;  tmp_59 = tmp_62 = None
        tmp_66 = tmp_64 + tmp_65;  tmp_64 = tmp_65 = None
        linear_4 = torch.nn.functional.linear(tmp_66, in_15, in_14);  tmp_66 = in_15 = in_14 = None
        tmp_68 = torch.nn.functional.layer_norm(linear_4, (256,), in_17, in_16, 1e-05);  linear_4 = in_17 = in_16 = None
        tmp_69 = torch.nn.functional.relu(tmp_68, inplace = True);  tmp_68 = None
        tmp_70 = tmp_69.reshape(1, 150, -1);  tmp_69 = None
        tmp_71 = tmp_70.permute(1, 0, 2);  tmp_70 = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_71, tmp_71, tmp_71, 512, 8, in_5, in_4, None, None, False, 0.0, in_3, in_2, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  in_5 = in_4 = in_3 = in_2 = None
        tmp_73 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_74 = torch.nn.functional.dropout(tmp_73, 0.0, False, False);  tmp_73 = None
        tmp_75 = torch.nn.functional.dropout(tmp_74, 0.0, False, False);  tmp_74 = None
        tmp_76 = tmp_71 + tmp_75;  tmp_71 = tmp_75 = None
        tmp_77 = torch.nn.functional.layer_norm(tmp_76, (512,), in_7, in_6, 1e-05);  tmp_76 = in_7 = in_6 = None
        tmp_78 = tmp_77.permute(1, 0, 2);  tmp_77 = None
        tmp_79 = tmp_78.reshape(1, 150, -1, 512);  tmp_78 = None
        linear_5 = torch.nn.functional.linear(tmp_79, in_11, in_10);  in_11 = in_10 = None
        tmp_81 = torch.nn.functional.relu(linear_5, inplace = True);  linear_5 = None
        tmp_82 = torch.nn.functional.dropout(tmp_81, 0.0, False, False);  tmp_81 = None
        return (tmp_82, tmp_79, tmp_32, conv2d_1)
        