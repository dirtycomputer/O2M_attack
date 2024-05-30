import csv

def clean_csv(input_filepath, output_filepath):
    # 打开输入文件
    with open(input_filepath, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)  # 读取标题行

        # 打开输出文件
        with open(output_filepath, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)  # 写入标题行
            
            for row in reader:
                # 处理每一行数据
                processed_row = [field.replace('\n', ' ') for field in row]  # 将字段中的换行符替换为空格
                writer.writerow(processed_row)  # 写入处理后的数据

# 使用示例
input_csv = 'RadFM-3MAD-Tiny-1K-respones.csv'
output_csv = 'RadFM-3MAD-Tiny-1K-respones1.csv'
clean_csv(input_csv, output_csv)
