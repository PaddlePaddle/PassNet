class Program_weight_tensor_meta_attention_scores:
	name = "in_0"
	original_name = "attention_scores"
	shape = [128, 6, 64, 64]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.000
	std = 0.054


class Program_weight_tensor_meta_extended_attention_mask_2:
	name = "in_1"
	original_name = "extended_attention_mask_2"
	shape = [128, 1, 1, 64]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.000
	std = 0.100


class Program_weight_tensor_meta_relative_position_scores_key:
	name = "in_2"
	original_name = "relative_position_scores_key"
	shape = [128, 6, 64, 64]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.000
	std = 0.002


class Program_weight_tensor_meta_relative_position_scores_query:
	name = "in_3"
	original_name = "relative_position_scores_query"
	shape = [128, 6, 64, 64]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.000
	std = 0.002
