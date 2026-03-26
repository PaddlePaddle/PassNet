class Program_weight_tensor_meta_extended_attention_mask:
	name = "in_0"
	original_name = "extended_attention_mask"
	shape = [32, 1, 64, 64]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = float("-inf")
	std = float("inf")


class Program_weight_tensor_meta_key_layer:
	name = "in_1"
	original_name = "key_layer"
	shape = [32, 12, 64, 32]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = 0.003
	std = 0.186


class Program_weight_tensor_meta_query_layer:
	name = "in_2"
	original_name = "query_layer"
	shape = [32, 12, 64, 32]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = 0.007
	std = 0.199


class Program_weight_tensor_meta_value_layer:
	name = "in_3"
	original_name = "value_layer"
	shape = [32, 12, 64, 32]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = -0.001
	std = 0.134
