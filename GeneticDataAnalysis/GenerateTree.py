import tsinfer

# 加载 .samples 文件
sample_data = tsinfer.load("ssa07.samples")

# 使用 tsinfer 进行树序列的推断
tree_sequence = tsinfer.infer(sample_data)

# 保存推断结果到一个 .trees 文件
tree_sequence.dump("ssa07.trees")

# 输出一些基本信息
print("推断的树序列已生成并保存为 'ssa02.trees'")
print("树的数量:", tree_sequence.num_trees)
print("序列长度:", tree_sequence.sequence_length)

