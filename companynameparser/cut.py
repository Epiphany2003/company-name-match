from companynameparser.parser import Parser
import pandas as pd

file_path = r"data/company.xlsx"
df = pd.read_excel(file_path)

par = Parser()
parser = par.parse

# 获取 'name' 列并初始化相关列表
names = df['name'].tolist()
place = []
suffix = []
brand = []
trade = []

# 解析每个公司名称并提取相关信息
for name in names:
    result = parser(name)
    place.append(result['place'])
    suffix.append(result['suffix'])
    brand.append(result['brand'])
    trade.append(result['trade'])

# 将解析的结果和原始的 'name' 合并到一个新的 DataFrame 中
new_df = pd.DataFrame({
    'name': names,
    'place': place,
    'suffix': suffix,
    'brand': brand,
    'trade': trade
})

# 保存到新的 CSV 文件
output_file = r"data/company_parsed.csv"
new_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"解析结果已保存到 {output_file}")
