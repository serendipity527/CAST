#!/usr/bin/env python3
"""
日志精简脚本 - 删除训练进度条输出

自动处理 logs/raw 目录下的所有日志文件，输出到 logs/clean 目录

用法: python clean_log.py
"""
import re
import sys
import os
from pathlib import Path

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
    
    try:
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
        
        return lines_removed, lines_kept
    except Exception as e:
        print(f"❌ 处理文件 {input_file} 时出错: {e}")
        return None, None

if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = Path("/home/dmx_MT/LZF/project/CAST/logs/raw")
    output_dir = Path("/home/dmx_MT/LZF/project/CAST/logs/clean")
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 获取所有日志文件
    log_files = list(input_dir.glob("*.log"))
    
    if not log_files:
        print(f"⚠️  在 {input_dir} 目录下未找到任何 .log 文件")
        sys.exit(0)
    
    print(f"📁 找到 {len(log_files)} 个日志文件")
    print(f"📂 输入目录: {input_dir}")
    print(f"📂 输出目录: {output_dir}")
    print("-" * 60)
    
    total_removed = 0
    total_kept = 0
    success_count = 0
    deleted_count = 0
    
    # 处理每个日志文件
    for log_file in log_files:
        output_file = output_dir / log_file.name
        print(f"\n处理: {log_file.name}")
        
        removed, kept = clean_log(log_file, output_file)
        
        if removed is not None and kept is not None:
            total_removed += removed
            total_kept += kept
            success_count += 1
            print(f"  ✅ 完成 - 删除: {removed:,} 行, 保留: {kept:,} 行")
            if removed + kept > 0:
                print(f"  📊 删除比例: {removed/(removed+kept)*100:.1f}%")
            
            # 处理成功后删除源文件
            try:
                log_file.unlink()
                deleted_count += 1
                print(f"  🗑️  已删除源文件: {log_file.name}")
            except Exception as e:
                print(f"  ⚠️  删除源文件失败: {e}")
        else:
            print(f"  ❌ 处理失败，保留源文件")
    
    print("\n" + "=" * 60)
    print(f"🎉 批量处理完成！")
    print(f"  - 成功处理: {success_count}/{len(log_files)} 个文件")
    print(f"  - 删除源文件: {deleted_count} 个")
    print(f"  - 总计删除: {total_removed:,} 行")
    print(f"  - 总计保留: {total_kept:,} 行")
    if total_removed + total_kept > 0:
        print(f"  - 总体删除比例: {total_removed/(total_removed+total_kept)*100:.1f}%")
