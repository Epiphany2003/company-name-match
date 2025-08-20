def extract_absolute_pure_data(input_file_path, output_file_path):
    """
    终极过滤：仅保留"纯公司名+空格+纯标签序列"，排除所有非格式内容（编号、前缀、说明等）
    :param input_file_path: 输入.txt文件路径
    :param output_file_path: 输出.txt文件路径
    """
    target_tags = {"B-ORG", "I-ORG", "O"}
    # 排除包含以下前缀/内容的行（覆盖编号、说明性前缀）
    exclude_prefixes = ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
                       "原始：", "原：", "示例：", "例如：", "分析：", "检查：", "修正：")
    extracted_lines = []

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                # 1. 排除带编号/说明前缀的行（如"1. 原始：北京爱贝佳..."）
                if any(stripped_line.startswith(prefix) for prefix in exclude_prefixes):
                    continue

                # 2. 分割行：过滤空字符，确保至少分为"公司名+标签"两部分
                parts = [p for p in stripped_line.split(' ') if p]
                if len(parts) < 2:
                    continue

                # 3. 验证标签序列：从后往前找，确保所有标签连续且为目标标签
                # （标签序列特征：末尾连续的B-ORG/I-ORG/O，前面为公司名）
                tag_count = 0
                # 从最后一个元素开始，统计连续的标签数量
                for part in reversed(parts):
                    if part in target_tags:
                        tag_count += 1
                    else:
                        break
                # 至少需要1个标签才是有效数据
                if tag_count == 0:
                    continue

                # 4. 提取纯公司名和纯标签序列（确保无多余内容）
                # 公司名：前半部分（总长度 - 标签数量）；标签序列：后半部分（标签数量）
                company_name_parts = parts[:-tag_count] if tag_count > 0 else parts
                tag_sequence = parts[-tag_count:] if tag_count > 0 else []
                # 公司名至少2个字符，标签序列至少1个标签
                if len(company_name_parts) < 1 or len(tag_sequence) < 1:
                    continue

                # 5. 重组为纯格式：公司名（空格连接） + 空格 + 标签序列（空格连接）
                pure_company_name = ' '.join(company_name_parts)
                pure_tag_sequence = ' '.join(tag_sequence)
                pure_line = f"{pure_company_name} {pure_tag_sequence}"

                # 6. 最终过滤：排除公司名含异常字符的行（如引号、括号内说明）
                if any(char in pure_company_name for char in ("\"", "“", "”", "（", "）", "(", ")")):
                    continue

                extracted_lines.append(pure_line)

        # 去重（避免重复数据）
        extracted_lines = list(set(extracted_lines))
        extracted_lines.sort()  # 可选：按公司名排序，便于查看

        # 写入输出
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(extracted_lines))

        print(f"终极处理完成！\n输入：{input_file_path}\n输出：{output_file_path}\n纯格式数据：{len(extracted_lines)} 条")

    except FileNotFoundError:
        print(f"错误：输入文件 '{input_file_path}' 未找到！")
    except Exception as e:
        print(f"错误：{str(e)}")


# ------------------- 配置文件路径 -------------------
if __name__ == "__main__":
    # 替换为你的原始文件路径（例："C:/hor_train_annotated.txt"）
    INPUT_FILE = "data/ner_data.txt"
    # 替换为输出文件路径（例："C:/final_pure_data.txt"）
    OUTPUT_FILE = "data/data_for_train.txt"
    
    extract_absolute_pure_data(INPUT_FILE, OUTPUT_FILE)