import tsinfer
import pandas as pd

# 加载表格数据
file_path = "C:\\Users\\86137\\Desktop\\tsinfer_try\\ssa07.csv" #输入表格路径
df = pd.read_csv(file_path)

# 创建一个新的 .samples 文件
with tsinfer.SampleData(path="ssa07_allpos.samples") as samples:
    # 添加 N 个 individual
    for _ in range(256):
        samples.add_individual()
    
    # 遍历每行，添加位点数据
    for _, row in df.iterrows():
        position = row[2]  # 选取 SNP 位置
        genotypes = row[3:].replace({2:1}).tolist()  # 其余列是基因型数据
        
        # 检查 genotypes 列表长度是否为 N
        if len(genotypes) == 256:
            samples.add_site(position=position, genotypes=genotypes)
        else:
            print(f"行 {_} 的基因型数据长度不为 256，实际长度为 {len(genotypes)}。")

print("Samples file created successfully as 'output.samples'")
