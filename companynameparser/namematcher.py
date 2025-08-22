import csv
import re
import opencc
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, AutoTokenizer
import torch
from companynameparser.parser import Parser

bank_mapping = {
    "工行": "中国工商银行",
    "农行": "中国农业银行",
    "中行": "中国银行",
    "建行": "中国建设银行",
    "交行": "交通银行",
    "邮储银行": "中国邮政储蓄银行",
    "招行": "招商银行",
    "民生银行": "中国民生银行",
    "浦发": "浦发银行",
    "中信": "中信银行",
    "光大": "中国光大银行",
    "华夏": "华夏银行",
    "广发": "广发银行",
    "兴业": "兴业银行",
    "平安": "平安银行",
    "恒丰": "恒丰银行",
    "浙商": "浙商银行",
    "渤海": "渤海银行",
    "徽商": "徽商银行",
    "农业银行": "中国农业银行",
    "工商银行": "中国工商银行",
    "建设银行": "中国建设银行",
    "农商行": "农村商业银行",
}

BERT_PATH = r'C:\Users\22403\Desktop\company-name-match\my-bert-base-chinese'
model = BertModel.from_pretrained(BERT_PATH)
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
model.eval()


def replace_bank_name(input_str):
    for short_name, full_name in bank_mapping.items():
        input_str = input_str.replace(short_name, full_name)
    return input_str


keywords = ["股份", "有限公司", "集团", "责任", "有限责任公司", "分行", "营业部", "控股", "股权", "投资", "贸易", "市"]
# keywords = []

def remove_keywords(input_str):
    pattern = "|".join(map(re.escape, keywords))
    result = re.sub(pattern, "", input_str)
    return result


def move_parentheses_content(input_str):
    pattern = r'[（(](.*?)[）)]'
    matches = re.findall(pattern, input_str)
    result = re.sub(pattern, '', input_str)
    if matches:
        result = ' '.join(matches) + ' ' + result
    return result.strip()


def traditional_to_simplified(input_str):
    converter = opencc.OpenCC('t2s')
    simplified_str = converter.convert(input_str)
    return simplified_str


def pre(str):
    str = traditional_to_simplified(str)
    str = replace_bank_name(str)
    str = move_parentheses_content(str)
    str = remove_keywords(str)
    return str.replace(" ", "")

def is_branch_company(suffix):
    # 定义分公司关键词列表
    branch_keywords = ["分行", "支行", "分公司", "营业部", "办事处", "分部", "分店", "代表处"]

    # 检查 suffix 是否包含任意分公司关键词
    for keyword in branch_keywords:
        if keyword in suffix:
            return True
    return False

def calculate_company_similarity(company1, company2):
    file_path = "C:\\Users\\22403\\Desktop\\company-name-match\\companynameparser\\data\\company.csv"
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        # 跳过表头
        header = next(reader, None)
        for row in reader:
            if (company1['name'] in row and company2['name'] in row) or (company2['name'] in row and company1['name'] in row):
                return 100

    if is_branch_company(company1['suffix']) or is_branch_company(company2['suffix']) or is_branch_company(company2['brand']) or is_branch_company(company1['brand']):
        return 0
    
    # 先检查是否为同品牌的分支机构
    is_branch1 = is_branch_company(company1['suffix'])
    is_branch2 = is_branch_company(company2['suffix'])
    
    # 如果品牌相同且至少有一个是分支机构，仍可计算相似度
    if (is_branch1 or is_branch2) and company1['brand'] != company2['brand']:
        return 0


    # 对行业进行BERT编码并计算余弦相似度
    industry1 = company1['trade']
    industry2 = company2['trade']

    if not industry1 and not industry2:  # 都为空时相似度100%
        industry_sim = 1.0

    inputs1 = tokenizer(industry1, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs2 = tokenizer(industry2, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    embedding1 = outputs1.pooler_output  # [1, hidden_size]
    embedding2 = outputs2.pooler_output  # [1, hidden_size]

    embedding1 = embedding1.numpy()
    embedding2 = embedding2.numpy()
    industry_sim = cosine_similarity(embedding1, embedding2)[0][0]

    company1_name = company1['brand']
    company2_name = company2['brand']

    if not company1_name and not company2_name:  # 都为空时相似度100%
        name_similarity = 1.0
    else:
    # 对公司名称进行编码
        name_similarity  = 0
        if company2_name != '' and company1_name != '':
            inputs1 = tokenizer(company1_name, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs2 = tokenizer(company2_name, return_tensors="pt", padding=True, truncation=True, max_length=128)

            # 使用 BERT 计算嵌入
            with torch.no_grad():
                outputs1 = model(**inputs1)
                outputs2 = model(**inputs2)

            # 提取池化后的句向量（BERT 的 [CLS] token 表示）
            embedding1 = outputs1.pooler_output  # [1, hidden_size]
            embedding2 = outputs2.pooler_output  # [1, hidden_size]

            # 转换为 NumPy 格式以便计算余弦相似度
            embedding1 = embedding1.numpy()
            embedding2 = embedding2.numpy()

        # 计算公司名称的余弦相似度
        name_similarity = cosine_similarity(embedding1, embedding2)[0][0]

    # 3. 地区相似度计算（新增，处理空值）
    place1 = company1['place']
    place2 = company2['place']
    if not place1 and not place2:  # 任一为空时相似度100%
        place_sim = 1.0
    else:
        inputs1 = tokenizer(place1, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs2 = tokenizer(place2, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs1 = model(** inputs1)
            outputs2 = model(**inputs2)
        embedding1 = outputs1.pooler_output.numpy()
        embedding2 = outputs2.pooler_output.numpy()
        place_sim = cosine_similarity(embedding1, embedding2)[0][0]

    # 4. 权重调整为：品牌8:行业1:地区1
    return name_similarity * 80 + industry_sim * 10 + place_sim * 10

def adjust_bank_to_trade(company):
    """将公司信息中后缀的“银行”迁移到行业字段"""
    suffix = company.get('suffix', '')
    if '银行' in suffix:
        # 将“银行”添加到行业字段（保留原有行业信息）
        current_trade = company.get('trade', '')
        company['trade'] = f"{current_trade}银行".strip()  # 避免空字符串时多空格
        
        # 从后缀中移除“银行”（处理可能的多余空格）
        company['suffix'] = suffix.replace('银行', '').strip()
    return company


if __name__ == "__main__":
    company1 = "深圳市腾讯科技有限公司"
    company2 = "腾讯"
    par = Parser()
    parser = par.parse
    com1 = parser(company1)
    com2 = parser(company2)
    similarity_score = calculate_company_similarity(com1, com2)
    print(f"公司1: {company1}")
    print(f"公司2: {company2}")
    print(f"匹配相似度分数: {similarity_score:.2f}")