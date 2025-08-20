import random

def random_sample(input_file, output_file, sample_size):
    """
    从输入文件中随机抽取指定数量的行并保存到输出文件
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        sample_size: 要抽取的样本数量
    """
    # 首先计算文件总行数
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # 确保样本量不超过总条数
    sample_size = min(sample_size, total_lines)
    print(f"从 {total_lines} 条数据中随机抽取 {sample_size} 条...")
    
    # 生成要抽取的随机行号（从0开始）
    selected_lines = set(random.sample(range(total_lines), sample_size))
    
    # 再次读取文件，提取选中的行
    with open(input_file, 'r', encoding='utf-8') as in_f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        for line_num, line in enumerate(in_f):
            if line_num in selected_lines:
                out_f.write(line)
                selected_lines.remove(line_num)  # 优化：减少后续检查时间
                if not selected_lines:  # 所有选中行都已处理，提前退出
                    break
    
    print(f"已完成！随机抽取的 {sample_size} 条数据已保存到 {output_file}")

if __name__ == "__main__":
    # 配置文件路径和要抽取的数量
    input_filename = "Company-Names-Corpus.txt"
    output_filename = "random_10000_company_names.txt"
    num_samples = 5000
    
    # 执行随机抽取
    random_sample(input_filename, output_filename, num_samples)
