import requests
import os
import time

# -------------------------- 1. 核心配置（根据需求修改） --------------------------
# 文件路径配置
SSH_KEY_PATH = "./ssh_key_suleidan"          # SSH密钥文件（同目录）
RAW_DATA_PATH = "data/ner_data.txt"            # 原始1W条数据文件
ANNOTATED_SAVE_PATH = "./data/hor_train_annotated.txt"  # 最终合并结果路径
TEMP_BATCH_PATH = "./temp_batch_results/"     # 临时批次结果保存目录

# 模型与批次配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
TARGET_MODEL = "deepseek-r1:32b"
BATCH_SIZE = 30  # 每批处理条目数（建议先测试50条）
TIMEOUT = 300     # 单批调用超时时间（秒）
RETRY_TIMES = 3    # 增加重试次数到3次

# -------------------------- 2. 工具函数（保持不变） --------------------------
def init_temp_dir():
    if not os.path.exists(TEMP_BATCH_PATH):
        os.makedirs(TEMP_BATCH_PATH)
        print(f"✅ 创建临时批次目录：{os.path.abspath(TEMP_BATCH_PATH)}")
    else:
        print(f"✅ 临时批次目录已存在：{os.path.abspath(TEMP_BATCH_PATH)}")

def read_and_split_data(file_path, batch_size):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"原始文件不存在：{file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
    total_lines = len(raw_lines)
    print(f"✅ 读取原始数据：共{total_lines}条有效条目")

    batches = []
    for i in range(0, total_lines, batch_size):
        batch_lines = raw_lines[i:i+batch_size]
        batches.append({
            "batch_num": i//batch_size + 1,
            "content": "\n".join(batch_lines),
            "line_count": len(batch_lines)
        })
    total_batches = len(batches)
    print(f"✅ 拆分完成：共{total_batches}批，每批{batch_size}条（最后一批{batches[-1]['line_count']}条）")

    completed_batches = []
    if os.path.exists(TEMP_BATCH_PATH):
        for file in os.listdir(TEMP_BATCH_PATH):
            if file.startswith("batch_") and file.endswith(".txt"):
                try:
                    batch_num = int(file.split("_")[1].split(".")[0])
                    completed_batches.append(batch_num)
                except:
                    continue
    if completed_batches:
        print(f"⚠️ 发现已完成批次：{sorted(completed_batches)}，将跳过这些批次")
    else:
        print(f"✅ 无已完成批次，将从第1批开始处理")

    return total_batches, batches, completed_batches

# -------------------------- 3. 优化：单批数据标注（强化约束+结果清洗） --------------------------
def annotate_single_batch(batch_info):
    batch_num = batch_info["batch_num"]
    batch_content = batch_info["content"]
    batch_line_count = batch_info["line_count"]

    # 优化1：强化Prompt，增加格式示例，明确条目数要求
    prompt = f"""
任务：重新检查以下第{batch_num}批NER数据（共{batch_line_count}条），仅标注「公司关键词」，严格遵循：

1. 标签规则：
   - 仅使用3种标签：B-ORG（公司关键词首字符）、I-ORG（公司关键词后续字符）、O（非关键词字符）；
   - 关键词定义：公司名中的核心名称，
   - 公司名称一般由地区（Region）、关键词（X）、行业（Industry）和公司后缀（Org_Suffix）四部分组成。比如【深圳市万网博通科技有限公司】，地区为【深圳市】、【万网博通】是关键词、【科技】是行业词，【有限公司是】公司后缀。我需要你先排除地区（包括国家、和中国的各个地方）、行业、常见公司后缀，剩下的部分就是关键词。
   - 只需要标出关键词和非关键词即可，但是关键词的定位通过先排除地区、行业、公司后缀来实现。

2. 格式强制要求（必须严格遵守，否则标注无效）：
   - 输出行数必须 = {batch_line_count}条（与输入条目数完全一致）；
   - 每条格式：公司名 + 空格 + 标签序列（如“阿里软件公司 B-ORG I-ORG O O O O”）；
   - 标签序列长度必须 = 公司名字符数（如“阿里”2字符 → 标签2个）；
   - 不允许添加任何额外内容（如解释、空行、批次说明、总结文字）。
   - 最后的输出只有符合格式的公司名 + 空格 + 标签序列

3. 错误示例（以下均为错误，禁止出现）：
   - 错误1：多输出一行“标注完成”；
   - 错误2：将“阿里软件公司”拆分为两行；
   - 错误3：标签序列长度与公司名不一致。

4. 正确示例（假设输入1条）：
   输入：阿里软件公司 O O O O O O
   输出：阿里软件公司 B-ORG I-ORG O O O O

第{batch_num}批原始数据：
{batch_content}

现在，请输出{batch_line_count}条标注结果（仅输出标注内容，无其他文字）：
"""

    # 调用模型（增加重试次数）
    for retry in range(RETRY_TIMES + 1):
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": TARGET_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.0,  # 优化2：温度设为0，最大限度降低随机性
                    "max_tokens": 50000
                },
                timeout=TIMEOUT
            )
            response.raise_for_status()
            raw_annotated = response.json()["response"].strip()

            # 优化3：结果清洗（过滤无效行，确保条目数匹配）
            # 步骤1：按行拆分，过滤空行和不含标签的行
            annotated_lines = []
            for line in raw_annotated.splitlines():
                line_clean = line.strip()
                # 仅保留包含有效标签的行（避免模型输出的解释文字）
                if "O" in line_clean or "B-ORG" in line_clean or "I-ORG" in line_clean:
                    annotated_lines.append(line_clean)
            
            # 步骤2：确保最终条目数与原始一致（截取或补充，极端情况处理）
            ''' if len(annotated_lines) > batch_line_count:
                # 若多标，取前N条（N=原始数量）
                #annotated_lines = annotated_lines[:batch_line_count]
                print(f"⚠️ 批次{batch_num}多标，已截取前{batch_line_count}条")
            elif len(annotated_lines) < batch_line_count:
                # 若少标，用原始行填充（避免后续合并失败，需人工检查）
                missing = batch_line_count - len(annotated_lines)
                raw_batch_lines = batch_content.splitlines()
                for i in range(missing):
                    # 填充原始行（未标注状态）
                    annotated_lines.append(raw_batch_lines[len(annotated_lines)] if len(annotated_lines) < len(raw_batch_lines) else "")
                print(f"⚠️ 批次{batch_num}少标，已用原始数据填充{missing}条（需人工检查）") '''
            
            # 重新拼接为文本
            annotated_content = "\n".join(annotated_lines)
            print(f"✅ 批次{batch_num}标注完成：{batch_line_count}条（清洗后）")
            return annotated_content

        except Exception as e:
            if retry < RETRY_TIMES:
                wait_time = (retry + 1) * 10
                print(f"⚠️ 批次{batch_num}第{retry+1}次失败：{str(e)}，{wait_time}秒后重试")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"批次{batch_num}重试{RETRY_TIMES}次仍失败：{str(e)}")

# -------------------------- 4. 保存与合并函数（保持不变） --------------------------
def save_batch_result(batch_num, annotated_content):
    batch_save_path = os.path.join(TEMP_BATCH_PATH, f"batch_{batch_num}.txt")
    with open(batch_save_path, "w", encoding="utf-8") as f:
        f.write(annotated_content)
    print(f"✅ 批次{batch_num}结果保存到：{batch_save_path}")

def merge_all_batches(total_batches, final_save_path):
    merged_content = []
    for batch_num in range(1, total_batches + 1):
        batch_path = os.path.join(TEMP_BATCH_PATH, f"batch_{batch_num}.txt")
        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"批次{batch_num}临时文件缺失：{batch_path}")
        with open(batch_path, "r", encoding="utf-8") as f:
            merged_content.append(f.read().strip())

    final_content = "\n".join(merged_content)
    with open(final_save_path, "w", encoding="utf-8") as f:
        f.write(final_content)

    raw_total = len([line.strip() for line in open(RAW_DATA_PATH, "r", encoding="utf-8").readlines() if line.strip()])
    final_total = len([line.strip() for line in final_content.splitlines() if line.strip()])
    print(f"\n🎉 所有批次合并完成！最终文件：{os.path.abspath(final_save_path)}")
    print(f"📊 总条目数验证：原始{raw_total}条 → 标注{final_total}条（{'匹配' if raw_total == final_total else '不匹配'}）")

# -------------------------- 5. 主流程 --------------------------
if __name__ == "__main__":
    try:
        init_temp_dir()
        total_batches, all_batches, completed_batches = read_and_split_data(RAW_DATA_PATH, BATCH_SIZE)

        for batch in all_batches:
            batch_num = batch["batch_num"]
            if batch_num in completed_batches:
                print(f"⏭️  跳过已完成批次：{batch_num}")
                continue

            print(f"\n⏳ 开始处理批次{batch_num}/{total_batches}（{batch['line_count']}条）")
            batch_annotated = annotate_single_batch(batch)
            save_batch_result(batch_num, batch_annotated)
            time.sleep(2)  # 批次间隔

        merge_all_batches(total_batches, ANNOTATED_SAVE_PATH)

    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        
        print("💡 提示：可重新运行脚本，将自动跳过已完成批次")
