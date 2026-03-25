class Program_weight_tensor_meta_attention_mask:
	name = "in_0"
	original_name = "attention_mask"
	shape = [1, 1, 21, 21]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = float("-inf")
	std = float("inf")


class Program_weight_tensor_meta_key:
	name = "in_1"
	original_name = "key"
	shape = [1, 4, 21, 4]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.001
	std = 0.014


class Program_weight_tensor_meta_query:
	name = "in_2"
	original_name = "query"
	shape = [1, 4, 21, 4]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.001
	std = 0.006


class Program_weight_tensor_meta_value:
	name = "in_3"
	original_name = "value"
	shape = [1, 4, 21, 4]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.000
	std = 0.015
