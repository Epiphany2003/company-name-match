import pandas as pd
import re

def extract_yes_no(text):
    """从文本中提取是否判断，返回'是'或'否'"""
    if not isinstance(text, str):
        return None  # 非字符串返回None表示无法判断
    
    # 转换为小写以便不区分大小写匹配
    text_lower = text.lower()
    
    # 检查是否包含肯定词汇
    yes_patterns = ['是同一家公司', '是同一公司', '是同一家', '是同一', '是']
    no_patterns = ['不是同一家公司', '不是同一公司', '不是同一家', '不是同一', '不是']
    
    # 优先检查更明确的模式
    for pattern in yes_patterns:
        if pattern in text_lower:
            return '是'
    
    for pattern in no_patterns:
        if pattern in text_lower:
            return '否'
    
    # 如果没有明确匹配，尝试用正则表达式查找单独的"是"或"否"
    if re.search(r'\b是\b', text_lower):
        return '是'
    if re.search(r'\b否\b', text_lower):
        return '否'
    
    # 如果都找不到，返回None表示无法判断
    return None

def calculate_accuracy(file_path):
    """计算大模型对比结果的准确率"""
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 检查必要的列是否存在
        required_columns = ['原始是否是同一家公司', '大模型对比结果']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Excel文件中缺少必要的列: {col}")
        
        # 提取大模型判断结果
        df['大模型判断提取'] = df['大模型对比结果'].apply(extract_yes_no)
        
        # 过滤掉无法判断的记录
        valid_data = df.dropna(subset=['原始是否是同一家公司', '大模型判断提取'])
        
        # 计算总有效样本数
        total_samples = len(valid_data)
        if total_samples == 0:
            print("没有有效的数据用于计算准确率")
            return
        
        # 计算准确的样本数
        correct_predictions = sum(
            valid_data['原始是否是同一家公司'] == valid_data['大模型判断提取']
        )
        
        # 计算准确率
        accuracy = correct_predictions / total_samples * 100
        
        # 输出结果
        print(f"总有效样本数: {total_samples}")
        print(f"判断正确的样本数: {correct_predictions}")
        print(f"准确率: {accuracy:.2f}%")
        
        # 可以选择将结果保存回Excel
        # df.to_excel('result_with_accuracy.xlsx', index=False)
        # print("结果已保存到'result_with_accuracy.xlsx'")
        
        return accuracy
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 替换为你的Excel文件路径
    excel_file_path = "company_similarity_final_with_accuracy.xlsx"
    calculate_accuracy(excel_file_path)
