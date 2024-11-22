import tskit

# 加载 .trees 文件
tree_sequence = tskit.load("ssa07.trees")

# 函数：将每棵树转换为括号表示
def tree_to_newick(tree, node):
    if tree.is_leaf(node):
        return str(node)
    else:
        children = tree.children(node)
        return "(" + ",".join(tree_to_newick(tree, child) for child in children) + ")"

# 打开一个 .txt 文件以写入树序列的括号表示
with open("ssa07.txt", "w") as file:
    file.write("Tree Sequence in Newick-like format:\n\n")
    
    # 遍历每棵树，并写入其括号表示
    for tree_index, tree in enumerate(tree_sequence.trees()):
        # 遍历每个根节点（以防有多个根节点的情况）
        root_representations = []
        for root in tree.roots:
            root_representations.append(tree_to_newick(tree, root))
        
        # 将每个根节点的表示连接为字符串
        newick_representation = ",".join(root_representations)
        file.write(f"Tree {tree_index}: {newick_representation}\n")

print("Tree sequence saved to 'tree_sequence.txt'")
