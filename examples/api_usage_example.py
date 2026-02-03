"""
API使用示例 - 展示如何使用HFSS API进行S参数提取和分析

本示例演示:
1. HFSS控制器的基本使用
2. S参数数据提取
3. 像素配置更新
4. 频率扫描和分析
5. 数据处理和可视化

基于test_api.py的使用方法，展示实际应用场景
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings
from typing import Dict, List, Tuple, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入API模块
from api import HFSSController

def setup_logging():
    """设置日志并屏蔽冗余输出"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # 降低第三方库日志级别，减少控制台噪声
    logging.getLogger('ansys').setLevel(logging.WARNING)
    logging.getLogger('pyaedt').setLevel(logging.WARNING)
    logging.getLogger('PyAEDT').setLevel(logging.WARNING)
    # 屏蔽 PyAEDT 的弃用警告
    warnings.filterwarnings('ignore', message='Method `data_real` is deprecated.*')
    warnings.filterwarnings('ignore', message='Method `data_imag` is deprecated.*')

def create_sample_pixel_configs():
    """创建示例像素配置 - 99个元素的列表形式"""
    import random
    
    # 生成99个像素值的示例配置
    configs = {
        'all_on': [2.0] * 99,  # 所有像素开启
        #'all_off': [0.001] * 99,  # 所有像素关闭
        #'random_pattern': [random.choice([0.001, 2.0]) for _ in range(99)],  # 随机模式
        #'alternating': [2.0 if i % 2 == 0 else 0.001 for i in range(99)],  # 交替模式
        #'center_focus': [2.0 if 40 <= i <= 58 else 0.001 for i in range(99)]  # 中心聚焦
    }
    return configs

def basic_api_usage():
    """基础API使用示例"""
    print("=" * 60)
    print("HFSS API 使用示例")
    print("=" * 60)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # HFSS项目路径 - 请根据实际情况修改
    PROJECT_PATH = r"C:\Users\16438\Desktop\python_HFSS\HFSS_Project\Project1.aedt"
    
    try:
        # 1. 检查项目文件是否存在
        print(f"\n1. 检查HFSS项目: {PROJECT_PATH}")
        if not os.path.exists(PROJECT_PATH):
            print(f"  警告: 项目文件不存在: {PROJECT_PATH}")
            print("  请确保项目路径正确，或创建新项目")
            return
        
        # 2. 创建HFSS控制器
        print("\n2. 初始化HFSS控制器...")
        hfss_ctrl = HFSSController(
            project_path=PROJECT_PATH,
            design_name="HFSSDesign1",  # 根据实际设计名称修改
            setup_name="Setup1",        # 根据实际设置名称修改
            sweep_name="Sweep"          # 根据实际扫频名称修改
        )
        print("  - HFSS控制器已创建")
        
        # 3. 连接到HFSS项目
        print(f"\n3. 连接到HFSS项目...")
        success = hfss_ctrl.connect()
        if success:
            print("  - 连接成功!")
        else:
            print("  - 连接失败，请检查HFSS是否运行和项目路径")
            return
            
        # 4. 定义频率范围
        print("\n4. 定义频率扫描范围...")
        freq_points = np.linspace(2e9, 7e9, 101)  # 2-7 GHz, 101个点
        print(f"  - 频率范围: {freq_points[0]/1e9:.1f} - {freq_points[-1]/1e9:.1f} GHz")
        print(f"  - 频率点数: {len(freq_points)}")
        
        # 5. 获取初始S参数
        print("\n5. 获取初始S参数...")
        try:
            initial_s11 = hfss_ctrl.get_s_params()
            if initial_s11 is not None and len(initial_s11) > 0:
                print(f"  - S11数据获取成功，数据点数: {len(initial_s11)}")
                # 假设返回的是DataFrame格式
                if hasattr(initial_s11, 'values'):
                    s11_values = initial_s11.values.flatten()
                else:
                    s11_values = initial_s11
                print(f"  - S11范围: {np.min(s11_values):.2f} 到 {np.max(s11_values):.2f} dB")
                
                # 找到谐振频率
                resonant_idx = np.argmin(s11_values)
                print(f"  - 主谐振点索引: {resonant_idx} (S11 = {s11_values[resonant_idx]:.2f} dB)")
            else:
                print("  - S11数据为空或获取失败")
                initial_s11 = None
            
        except Exception as e:
            print(f"  - S参数获取失败: {str(e)}")
            initial_s11 = None
            
        # 6. 测试不同像素配置
        print("\n6. 测试不同像素配置...")
        
        pixel_configs = create_sample_pixel_configs()
        results = {}
        
        for config_name, pixel_config in pixel_configs.items():
            print(f"\n  测试配置: {config_name}")
            print(f"    像素配置: k = [{pixel_config[0]}, {pixel_config[1]}, ..., {pixel_config[-1]}] (共{len(pixel_config)}个)")
                
            try:
                # 使用set_variable方法设置k参数
                print(f"    更新像素配置...")
                success = hfss_ctrl.set_variable("k", pixel_config, unit="mm")
                if not success:
                    print(f"    像素配置更新失败")
                    continue
                
                # 运行分析
                print(f"    运行HFSS分析...")
                analyze_success = hfss_ctrl.analyze()
                if not analyze_success:
                    print(f"    分析失败")
                    continue
                    
                # 获取S参数
                s11_data = hfss_ctrl.get_s_params()
                
                if s11_data is not None:
                    # 处理数据格式
                    if hasattr(s11_data, 'values'):
                        s11_values = s11_data.values.flatten()
                    else:
                        s11_values = s11_data
                    
                    print(f"    原始S11数据类型: {type(s11_values)}")
                    print(f"    原始S11数据长度: {len(s11_values)}")
                    print(f"    是否复数: {np.iscomplexobj(s11_values)}")
                    print(f"    前5个原始值: {s11_values[:5]}")
                    
                    # 转换复数为dB - 这是关键步骤
                    if np.iscomplexobj(s11_values):
                        s11_db = 20 * np.log10(np.abs(s11_values))
                        print(f"    转换为dB后前5个值: {s11_db[:5]}")
                    else:
                        # 如果已经是实数，检查是否已经是dB值
                        if np.all(s11_values <= 0):  # dB值通常是负数
                            s11_db = s11_values
                            print("    数据似乎已经是dB值")
                        else:
                            # 假设是线性值，转换为dB
                            s11_db = 20 * np.log10(np.abs(s11_values))
                            print(f"    线性值转换为dB后前5个值: {s11_db[:5]}")
                    
                    # 分析结果
                    min_s11 = np.min(s11_db)
                    min_idx = np.argmin(s11_db)
                    
                    # 计算带宽 (S11 < -10 dB)
                    bandwidth_mask = s11_db < -10.0
                    if np.any(bandwidth_mask):
                        bandwidth_count = np.sum(bandwidth_mask)
                    else:
                        bandwidth_count = 0
                        
                    results[config_name] = {
                        's11_data': s11_db,  # 保存dB值而不是原始值
                        'min_s11': min_s11,
                        'min_index': min_idx,
                        'bandwidth_points': bandwidth_count
                    }
                    
                    print(f"    结果:")
                    print(f"      - 最小S11: {min_s11:.2f} dB")
                    print(f"      - 最小S11索引: {min_idx}")
                    print(f"      - 10dB带宽点数: {bandwidth_count}")
                else:
                    print(f"    S参数获取失败")
                
            except Exception as e:
                print(f"    测试失败: {str(e)}")
                
        # 7. 保存CSV后再从CSV绘图
        print("\n7. 保存CSV并从CSV绘图...")
        
        # 先保存每个配置的S参数CSV（完整和简化版本）
        for config_name, pixel_config in pixel_configs.items():
            print(f"\n  保存配置 {config_name} 的S参数CSV...")
            try:
                hfss_ctrl.set_variable("k", pixel_config, unit="mm")
                analyze_success = hfss_ctrl.analyze()
                if not analyze_success:
                    print("    分析失败，跳过保存")
                    continue
                s_params_df = hfss_ctrl.get_s_params()
                if s_params_df is None or s_params_df.empty:
                    print("    未获取到S参数数据，跳过保存")
                    continue
                # 保存完整CSV（包含原始列）
                full_csv = f"api_usage_s11_{config_name}.csv"
                saved_path = hfss_ctrl.save_s_params(s_params_df, full_csv)
                if saved_path:
                    print(f"    ✅ 已保存: {saved_path}")
                else:
                    print("    ❌ 保存完整CSV失败")
                # 保存简化CSV（Freq [GHz], dB(S(1,1)) []）
                if 'Freq' in s_params_df.columns and 'S(1,1)' in s_params_df.columns:
                    simple_df = pd.DataFrame({
                        'Freq [GHz]': s_params_df['Freq'] / 1e9,
                        'dB(S(1,1)) []': 20 * np.log10(np.abs(s_params_df['S(1,1)']))
                    })
                    simple_csv = f"api_usage_s11_simple_{config_name}.csv"
                    simple_df.to_csv(simple_csv, index=False)
                    print(f"    ✅ 已保存简化CSV: {simple_csv}")
                else:
                    print("    ⚠️ 数据列缺失，无法生成简化CSV (需要 'Freq' 和 'S(1,1)')")
            except Exception as e:
                print(f"    ❌ 保存失败: {str(e)}")
        
        # 从CSV收集数据用于绘图（优先使用简化CSV）
        plot_data = {}
        for config_name in pixel_configs.keys():
            try:
                simple_csv = f"api_usage_s11_simple_{config_name}.csv"
                full_csv = f"api_usage_s11_{config_name}.csv"
                if os.path.exists(simple_csv):
                    df = pd.read_csv(simple_csv)
                    if 'Freq [GHz]' in df.columns and 'dB(S(1,1)) []' in df.columns:
                        frequencies_ghz = df['Freq [GHz]']
                        s11_db = df['dB(S(1,1)) []']
                        plot_data[config_name] = {
                            'frequencies_ghz': frequencies_ghz,
                            's11_db': s11_db
                        }
                        print(f"✅ 从简化CSV加载: {simple_csv}")
                    else:
                        print(f"⚠️ 简化CSV格式不正确: {simple_csv}")
                elif os.path.exists(full_csv):
                    df = pd.read_csv(full_csv)
                    # 兼容完整CSV（支持多种列名与单位）
                    # 1) 选择频率列并进行单位检测
                    freq_col = None
                    for cand in ['Freq [GHz]', 'Frequency [GHz]', 'Frequency', 'Freq', 'freq']:
                        if cand in df.columns:
                            freq_col = cand
                            break
                    if freq_col is None:
                        print(f"⚠️ 未找到频率列，跳过: {full_csv}")
                        continue
                    freq_vals = pd.to_numeric(df[freq_col], errors='coerce')
                    if np.nanmax(freq_vals) > 1e6:
                        frequencies_ghz = freq_vals / 1e9
                    else:
                        frequencies_ghz = freq_vals
                    # 2) 优先使用 dB 列，其次解析复数列
                    db_cols = [c for c in df.columns if ('dB' in c and ('S(' in c or 'S11' in c))]
                    if len(db_cols) > 0:
                        s11_db = pd.to_numeric(df[db_cols[0]], errors='coerce')
                    else:
                        complex_cols = [c for c in df.columns if ('S(' in c or 'S11' in c)]
                        if len(complex_cols) == 0:
                            print(f"⚠️ 未找到S11列，跳过: {full_csv}")
                            continue
                        s11_raw = df[complex_cols[0]].astype(str).str.strip().str.replace('[()]', '', regex=True).str.replace(' ', '', regex=False)
                        try:
                            s11_complex = s11_raw.apply(lambda x: complex(x) if 'j' in x else complex(float(x)))
                            s11_db = 20 * np.log10(np.abs(s11_complex))
                        except Exception:
                            s11_numeric = pd.to_numeric(s11_raw, errors='coerce')
                            s11_db = 20 * np.log10(np.abs(s11_numeric))
                    plot_data[config_name] = {
                        'frequencies_ghz': frequencies_ghz,
                        's11_db': s11_db
                    }
                    print(f"✅ 从完整CSV加载: {full_csv}")
                else:
                    print(f"❌ 未找到CSV: {simple_csv} 或 {full_csv}")
            except Exception as e:
                print(f"❌ 加载绘图数据失败: {str(e)}")
        
        if plot_data:
            print("\n生成S11图表(基于CSV)...")
            fig, (ax_main, ax_sub) = plt.subplots(2, 1, figsize=(12, 8))
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            # 主图 - S11 dB曲线
            for i, (config_name, data) in enumerate(plot_data.items()):
                ax_main.plot(data['frequencies_ghz'], data['s11_db'], 'o-', 
                            linewidth=2, label=f'S11 ({config_name})', 
                            color=colors[i % len(colors)], markersize=3)
                min_idx = np.argmin(data['s11_db'])
                min_freq = data['frequencies_ghz'][min_idx]
                min_val = data['s11_db'][min_idx]
                ax_main.annotate(f'{config_name}\n{min_val:.2f} dB @ {min_freq:.2f} GHz',
                                xy=(min_freq, min_val),
                                xytext=(min_freq + 0.3, min_val + 3),
                                arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)]),
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                                fontsize=8)
            ax_main.set_xlabel('频率 (GHz)')
            ax_main.set_ylabel('S11 (dB)')
            ax_main.set_title('像素天线S11参数对比 (CSV)')
            ax_main.grid(True, alpha=0.3)
            ax_main.legend()
            ax_main.axhline(y=-10, color='r', linestyle='--', alpha=0.7, label='-10 dB参考线')
            # 子图 - 第一个配置的幅度
            first_config = list(plot_data.keys())[0]
            first_data = plot_data[first_config]
            s11_magnitude = 10**(first_data['s11_db']/20)
            ax_sub.plot(first_data['frequencies_ghz'], s11_magnitude, 'g-', 
                        linewidth=2, label=f'|S11| ({first_config})', marker='s', markersize=2)
            ax_sub.set_xlabel('频率 (GHz)')
            ax_sub.set_ylabel('幅度')
            ax_sub.set_title(f'S11幅度 ({first_config})')
            ax_sub.grid(True, alpha=0.3)
            ax_sub.legend()
            plt.tight_layout()
            chart_path = 'api_usage_s11_chart.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ S11图表已保存到: {chart_path}")
            # 简化图表（CSV）
            plt.figure(figsize=(10, 6))
            for i, (config_name, data) in enumerate(plot_data.items()):
                plt.plot(data['frequencies_ghz'], data['s11_db'], 'o-', 
                        linewidth=2, label=f'S11 ({config_name})', 
                        color=colors[i % len(colors)], markersize=4)
            plt.axhline(y=-10, color='r', linestyle='--', alpha=0.7, label='-10 dB参考线')
            plt.xlabel('频率 (GHz)')
            plt.ylabel('S11 (dB)')
            plt.title('S11参数 (dB, 基于CSV)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            simple_chart_path = 'api_usage_s11_simple.png'
            plt.savefig(simple_chart_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ 简化S11图表已保存到: {simple_chart_path}")
        else:
            print("❌ 没有数据可用于绘图")
            
        # 7. 保存数据
        print("\n7. 保存分析数据...")
        
        if results:
            # 创建DataFrame
            summary_data = []
            for config_name, data in results.items():
                # 计算谐振频率（最小S11对应的频率）
                min_idx = data['min_index']
                if min_idx < len(freq_points):
                    resonant_freq = freq_points[min_idx]
                else:
                    resonant_freq = freq_points[0]  # 默认值
                    
                summary_data.append({
                    'Configuration': config_name,
                    'Min_S11_dB': data['min_s11'],
                    'Resonant_Freq_GHz': resonant_freq / 1e9,
                    'Bandwidth_10dB_Points': data['bandwidth_points']
                })
                
            df_summary = pd.DataFrame(summary_data)
            
            # 保存CSV
            csv_path = 'api_usage_summary.csv'
            df_summary.to_csv(csv_path, index=False)
            print(f"  - 摘要数据已保存: {csv_path}")
            
            # 保存详细的S参数数据到CSV
            print("\n保存S参数数据...")
            
            for config_name, pixel_config in pixel_configs.items():
                print(f"\n处理配置: {config_name}")
                
                try:
                    # 设置像素配置
                    hfss_ctrl.set_variable("k", pixel_config, unit="mm")
                    
                    # 运行分析
                    hfss_ctrl.analyze()
                    
                    # 获取S参数
                    s_params_df = hfss_ctrl.get_s_params()
                    
                    if s_params_df is not None and not s_params_df.empty:
                        # 保存使用API的save_s_params
                        csv_filename = f'api_usage_s11_{config_name}.csv'
                        saved_path = hfss_ctrl.save_s_params(s_params_df, csv_filename)
                        
                        if saved_path:
                            print(f"✅ S参数数据已保存到: {saved_path}")
                            
                            # 额外处理为用户期望的格式（Freq [GHz], dB(S(1,1)) []）
                            # 假设s_params_df有'Freq'和'S(1,1)'列
                            if 'Freq' in s_params_df.columns and 'S(1,1)' in s_params_df.columns:
                                simple_df = pd.DataFrame({
                                    'Freq [GHz]': s_params_df['Freq'] / 1e9,
                                    'dB(S(1,1)) []': 20 * np.log10(np.abs(s_params_df['S(1,1)']))
                                })
                                simple_csv = f'api_usage_s11_simple_{config_name}.csv'
                                simple_df.to_csv(simple_csv, index=False)
                                print(f"✅ 简化格式保存到: {simple_csv}")
                            else:
                                print("⚠️ 数据格式不匹配，无法创建简化CSV")
                        else:
                            print("❌ 保存失败")
                    else:
                        print("❌ 未获取到S参数数据")
                        
                except Exception as e:
                    print(f"❌ 处理配置失败: {str(e)}")
            
            print("\nS参数数据保存完成")
            print(f"\n  分析摘要:")
            print(df_summary.to_string(index=False))
            
        print("\n" + "=" * 60)
        print("API使用示例完成!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"API使用过程中出现错误: {str(e)}")
        print(f"\n错误: {str(e)}")
        print("请检查:")
        print("1. HFSS是否正在运行")
        print("2. 项目文件是否存在且可访问")
        print("3. HFSS项目是否包含正确的天线模型")
        raise
    
    finally:
        # 清理资源
        try:
            if 'hfss_ctrl' in locals():
                # 这里可以添加清理代码，如关闭HFSS连接
                pass
        except:
            pass

def frequency_sweep_example():
    """频率扫描示例"""
    print("\n" + "=" * 60)
    print("频率扫描分析示例")
    print("=" * 60)
    
    PROJECT_PATH = r"C:\Users\16438\Desktop\python_HFSS\pixel_antenna_project.aedt"
    
    try:
        hfss_ctrl = HFSSController()
        
        if not hfss_ctrl.connect(PROJECT_PATH):
            print("无法连接到HFSS项目")
            return
            
        # 定义多个频率范围进行扫描
        frequency_bands = {
            'WiFi_2.4G': (2.3e9, 2.5e9, 21),
            'WiFi_5G': (5.1e9, 5.9e9, 41),
            'Sub6_5G': (3.3e9, 3.8e9, 26)
        }
        
        print("\n扫描不同频段:")
        
        for band_name, (f_start, f_stop, n_points) in frequency_bands.items():
            print(f"\n  {band_name}: {f_start/1e9:.1f} - {f_stop/1e9:.1f} GHz")
            
            freq_points = np.linspace(f_start, f_stop, n_points)
            
            try:
                s11_data = hfss_ctrl.get_s_params(freq_points)
                
                # 分析该频段
                min_s11 = np.min(s11_data)
                min_freq = freq_points[np.argmin(s11_data)]
                avg_s11 = np.mean(s11_data)
                
                print(f"    - 最小S11: {min_s11:.2f} dB @ {min_freq/1e9:.2f} GHz")
                print(f"    - 平均S11: {avg_s11:.2f} dB")
                
                # 检查是否满足常见标准
                good_points = np.sum(s11_data < -10.0)
                coverage = good_points / len(s11_data) * 100
                print(f"    - S11 < -10dB 覆盖率: {coverage:.1f}%")
                
            except Exception as e:
                print(f"    - 扫描失败: {str(e)}")
                
    except Exception as e:
        print(f"频率扫描失败: {str(e)}")

if __name__ == "__main__":
    print("HFSS API 使用示例")
    print("\n选择运行模式:")
    print("1. 基础API使用示例")
    print("2. 频率扫描示例")
    print("3. 运行所有示例")
    
    try:
        choice = input("\n请输入选择 (1, 2, 或 3): ").strip()
        
        if choice == "1":
            basic_api_usage()
        elif choice == "2":
            frequency_sweep_example()
        elif choice == "3":
            basic_api_usage()
            frequency_sweep_example()
        else:
            print("无效选择，运行基础示例...")
            basic_api_usage()
            
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n执行失败: {str(e)}")
        import traceback
        traceback.print_exc()