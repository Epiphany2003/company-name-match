import pandas as pd
import requests
import json
import time
from companynameparser.parser import Parser
from companynameparser.namematcher import calculate_company_similarity, adjust_bank_to_trade

# 配置大模型参数
SSH_KEY_PATH = ".bertTrain/ssh_key_suleidan"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
TARGET_MODEL = "deepseek-r1:32b"

def call_llm(company1, company2, retry_times=3):  # 新增重试参数
    """调用大模型判断两家公司是否为同一家，返回'是'或'否'（带重试）"""
    prompt = f"""请判断以下两家公司是否为同一家公司，仅返回'是'或'否'，不要其他内容。
公司1: {company1}
公司2: {company2}"""
    
    payload = {
        "model": TARGET_MODEL,
        "prompt": prompt,
        "stream": False,
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    # 重试逻辑
    for retry in range(retry_times + 1):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
            response.raise_for_status()  # 若状态码非200，抛异常
            result = response.json()
            llm_output = result.get("response", "").strip()
            
            # 修改后的逻辑
            if "是" in llm_output:
                return "是"
            elif "否" in llm_output:
                return "否"
            else:
                print(f"大模型返回非预期结果: {llm_output}，默认返回'否'")
                return "否"
        
        except Exception as e:
            if retry < retry_times:
                wait_time = (retry + 1) * 10  # 重试间隔：10s、20s、30s...
                print(f"调用大模型第{retry+1}次失败: {str(e)}，{wait_time}秒后重试")
                time.sleep(wait_time)
            else:
                # 重试次数用尽，返回默认值
                print(f"调用大模型重试{retry_times}次均失败: {str(e)}，默认返回'否'")
                return "否"

def main():
    # 读取Excel文件
    file_path = "similarity_pairs.xlsx"
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
        original_judgment = row.iloc[2] if pd.notna(row.iloc[2]) else None
        llm_result = None  # 重置大模型结果，将根据score动态生成
        
        try:
            # 解析两家公司的信息
            com1 = parser(company1)
            com1 = adjust_bank_to_trade(com1)
            com2 = parser(company2)
            com2 = adjust_bank_to_trade(com2)
            
            # 计算相似度分数
            score = calculate_company_similarity(com1, com2)
            
            # 判断逻辑：score为0时调用大模型，否则用代码判断
            if score == 0:
                code_judgment = None  # 代码不判断，由大模型决定
                llm_result = call_llm(company1, company2)
                final_judgment = llm_result
            else:
                code_judgment = "是" if score >= 80 else "否"
                final_judgment = code_judgment
            
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
                "代码计算分数": round(score, 2) if score != 0 else "触发大模型判断",
                "代码判断是否是一家公司": code_judgment,
                "原始是否是同一家公司": original_judgment,
                "大模型对比结果": llm_result,
                "最终判断结果": final_judgment  # 新增最终判断列
            }
            results.append(result)
            
            # 计算准确率（只统计原始判断不为空的行）
            if original_judgment is not None:
                total += 1
                if str(final_judgment).strip() == str(original_judgment).strip():
                    correct += 1
            
        except Exception as e:
            print(f"处理第{index+1}行时出错: {e}")
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
                "大模型对比结果": None,
                "最终判断结果": None
            })
    
    # 转换为DataFrame并保持列顺序
    result_df = pd.DataFrame(results)
    
    # 计算并显示准确率
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n准确率统计:")
        print(f"总有效样本数: {total}")
        print(f"判断正确的样本数: {correct}")
        print(f"综合判断准确率: {accuracy:.2f}%")
        
        # 添加统计行
        summary_row = pd.Series({
            "公司1": f"准确率: {accuracy:.2f}%",
            "公司1地区": f"正确/总样本: {correct}/{total}",
            "最终判断结果": "统计结果"
        }, name="统计信息")
        result_df = pd.concat([result_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # 保存结果
    output_path = "company_similarity_final_with_accuracy.xlsx"
    result_df.to_excel(output_path, index=False)
    print(f"\n处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    main()