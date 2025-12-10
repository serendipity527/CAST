#!/usr/bin/env python3
"""
日志精简脚本 - 删除训练进度条输出

用法: python clean_log.py <文件名>
示例: python clean_log.py train_etth1_gpt2_reason.log
      输入: train_etth1_gpt2_reason.log
      输出: output_log/train_etth1_gpt2_reason.log
"""
import re
import sys
import os

def clean_log(input_file, output_file):
    """
    删除日志中的tqdm进度条输出
    
    匹配模式：
    - 805it [00:38, 20.32it/s]
    - 0it [00:00, ?it/s]
    - 1it [00:00, 1.58it/s]
    """
    # 匹配进度条格式的正则表达式
    # 格式: 数字 + it [时:分, 速度it/s]
    progress_pattern = re.compile(r'^\d+it \[\d+:\d+,\s*[\d.?]+it/s\]')
    
    lines_removed = 0
    lines_kept = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                # 检查是否是进度条行
                if progress_pattern.match(line.strip()):
                    lines_removed += 1
                    continue
                
                # 保留其他所有行
                f_out.write(line)
                lines_kept += 1
    
    print(f"✅ 日志精简完成！")
    print(f"  - 输入文件: {input_file}")
    print(f"  - 输出文件: {output_file}")
    print(f"  - 删除行数: {lines_removed:,}")
    print(f"  - 保留行数: {lines_kept:,}")
    print(f"  - 删除比例: {lines_removed/(lines_removed+lines_kept)*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python clean_log.py <文件名>")
        print("示例: python clean_log.py train_etth1_gpt2_reason.log")
        sys.exit(1)
    
    filename = sys.argv[1]
    input_log = filename
    output_log = os.path.join("logs", filename)
    
    # 确保输出目录存在
    os.makedirs("output_log", exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_log):
        print(f"❌ 错误: 输入文件不存在: {input_log}")
        sys.exit(1)
    
    clean_log(input_log, output_log)
