import pandas as pd
from companynameparser.parser import Parser
from companynameparser.namematcher import calculate_company_similarity, adjust_bank_to_trade

def main():
    # 读取Excel文件
    file_path = "similarity_pairs.xlsx"  # 替换为你的输入文件路径
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    
    # 初始化解析器
    par = Parser()
    parser = par.parse
    
    # 存储结果的列表
    results = []
    # 用于计算准确率的计数器
    total = 0
    correct = 0
    
    # 遍历每行数据
    for index, row in df.iterrows():
        # 读取输入文档的前四列数据
        company1 = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        company2 = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
        original_judgment = row.iloc[2] if pd.notna(row.iloc[2]) else None  # 原始是否同一家公司（M列）
        llm_result = row.iloc[3] if pd.notna(row.iloc[3]) else None         # 大模型对比结果
        
        try:
            # 解析两家公司的信息
            com1 = parser(company1)
            com1 = adjust_bank_to_trade(com1)
            com2 = parser(company2)
            com2 = adjust_bank_to_trade(com2)
            
            # 计算相似度分数
            score = calculate_company_similarity(com1, com2)
            # 判断是否为同一家公司（80分及以上）
            code_judgment = "是" if score >= 80 else "否"
            
            # 按照指定列顺序存储结果
            result = {
                "公司1": company1,
                "公司1地区": com1["place"],
                "公司1品牌": com1["brand"],
                "公司1行业": com1["trade"],
                "公司1后缀": com1["suffix"],
                "公司2": company2,
                "公司2地区": com2["place"],
                "公司2品牌": com2["brand"],
                "公司2行业": com2["trade"],
                "公司2后缀": com2["suffix"],
                "代码计算分数": round(score, 2),
                "代码判断是否是一家公司": code_judgment,
                "原始是否是同一家公司": original_judgment,
                "大模型对比结果": llm_result
            }
            results.append(result)
            
            # 计算准确率（只统计原始判断不为空的行）
            if original_judgment is not None:
                total += 1
                # 统一判断标准的格式（处理可能的大小写或空格问题）
                if str(code_judgment).strip() == str(original_judgment).strip():
                    correct += 1
            
        except Exception as e:
            print(f"处理第{index+1}行时出错: {e}")
            # 出错时填充空值
            results.append({
                "公司1": company1,
                "公司1地区": None,
                "公司1品牌": None,
                "公司1行业": None,
                "公司1后缀": None,
                "公司2": company2,
                "公司2地区": None,
                "公司2品牌": None,
                "公司2行业": None,
                "公司2后缀": None,
                "代码计算分数": None,
                "代码判断是否是一家公司": None,
                "原始是否是同一家公司": original_judgment,
                "大模型对比结果": llm_result
            })
    
    # 转换为DataFrame并保持列顺序
    result_df = pd.DataFrame(results)
    
    # 计算并显示准确率
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n准确率统计:")
        print(f"总有效样本数: {total}")
        print(f"判断正确的样本数: {correct}")
        print(f"代码判断准确率: {accuracy:.2f}%")
        
        # 在结果中添加准确率信息（作为最后一行）
        summary_row = pd.Series({
            "公司1": f"准确率: {accuracy:.2f}%",
            "公司1地区": f"正确/总样本: {correct}/{total}",
            "代码判断是否是一家公司": "统计结果"
        }, name="统计信息")
        result_df = pd.concat([result_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # 保存结果到新Excel文件
    output_path = "company_similarity_final_with_accuracy.xlsx"
    result_df.to_excel(output_path, index=False)
    print(f"\n处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    main()
