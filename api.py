"""
HFSS API 接口库 - 使用 PyAEDT 实现 Ansys HFSS 的自动化控制
主要功能：变量修改、运行仿真、获取 S 参数结果、获取远场数据
设计原则：简洁性、稳定性、可维护性
"""
import os
import time
import psutil
import traceback
import numpy as np
import pandas as pd
from ansys.aedt.core import Hfss
import time

class HFSSController:
    """HFSS 自动化控制接口
    
    通过上下文管理器管理 HFSS 会话生命周期，确保资源正确释放：
    with HFSSController(...) as hfss:
        # 使用 hfss 对象
    """
    
    def __init__(self, project_path, design_name="HFSSDesign1", 
                 setup_name="Setup1", sweep_name="Sweep", port=54100,
                 default_length_unit='mm', default_angle_unit="deg"):
        """
        初始化 HFSS 控制器
        
        :param project_path: HFSS 项目路径 (.aedt)
        :param design_name: 设计名称 (默认: "HFSSDesign1")
        :param setup_name: 仿真设置名称 (默认: "Setup1")
        :param sweep_name: 扫频名称 (默认: "Sweep")
        :param port: gRPC 端口 (默认: 54100)
        :param default_length_unit: 默认长度单位 (默认: "mm")
        :param default_angle_unit: 默认角度单位 (默认: "deg")
        """
        self.project_path = project_path
        self.lock_file = project_path + ".lock"
        self.design_name = design_name
        self.setup_name = setup_name
        self.sweep_name = sweep_name
        self.port = port
        self.default_length_unit = default_length_unit
        self.default_angle_unit = default_angle_unit
        self.hfss = None
        self._desktop = None
        self.model_units = None  # 存储模型单位
    
    def _force_unlock_file(self, file_path):
        """强制解除文件锁定
       
        当检测到锁文件时，尝试终止占用进程并删除锁文件
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ 已清除锁文件: {file_path}")
                return True
        except PermissionError:
            print("⚠️ 尝试终止占用进程...")
            for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                try:
                    # 查找占用锁文件的 ANSYS 进程
                    if "ansysedt.exe" in proc.info['name'].lower():
                        for file in proc.info.get('open_files', []):
                            if file_path.lower() in file.path.lower():
                                print(f"终止进程: PID={proc.pid}, 名称={proc.info['name']}")
                                proc.kill()
                                time.sleep(2)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                    continue
            print("❌ 删除失败：请重启电脑后手动删除锁文件")
        except Exception as e:
            print(f"❌ 解锁文件错误: {str(e)}")
        return False
    
    def connect(self):
        """连接到 HFSS 并打开项目

        返回: True 连接成功, False 连接失败
        """
        try:
            # 清除可能存在的锁文件
            if os.path.exists(self.lock_file):
                print("⚠️ 检测到锁文件，尝试清除...")
                self._force_unlock_file(self.lock_file)
            
            # 创建 HFSS 会话
            print("🚀 启动 HFSS 会话...")
            self.hfss = Hfss(
                project=self.project_path,
                design=self.design_name,
                version="2023.1",
                new_desktop=True,
                close_on_exit=False,
                port=self.port
            )
            self._desktop = self.hfss._desktop
            
            # 获取并存储模型单位
            self.model_units = self.hfss.modeler.model_units
            print(f"🔗 已连接项目: {self.hfss.project_name} (单位: {self.model_units})")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def check_design_config(self):
        """检查设计配置是否有效
        
        验证 setup 和 sweep 是否存在
        返回: True 配置有效, False 配置无效
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            print("\n📋 设计配置检查:")
            
            # 1. 检查 Setup 是否存在
            setup_names = [setup.name for setup in self.hfss.setups]
            print(f"  可用 Setup 列表: {setup_names}")
            if self.setup_name not in setup_names:
                raise ValueError(f"❌ 未找到 Setup: {self.setup_name}（可用：{setup_names}）")
            
            # 2. 检查 Sweep 是否存在
            setup = self.hfss.get_setup(self.setup_name)
            if not setup:
                raise ValueError(f"❌ 无法获取 Setup 对象: {self.setup_name}")
            
            sweep_names = [sweep.name for sweep in setup.sweeps]
            print(f"  {self.setup_name} 下的 Sweep 列表: {sweep_names}")
            
            # 更新扫频名称（如果找不到则使用第一个）
            if sweep_names:
                if self.sweep_name not in sweep_names:
                    print(f"⚠️ 未找到指定 Sweep: {self.sweep_name}，使用第一个可用 Sweep: {sweep_names[0]}")
                    self.sweep_name = sweep_names[0]
            else:
                print("⚠️ 未找到任何 Sweep，将直接使用 Setup")
                self.sweep_name = None
            
            return True
        except Exception as e:
            print(f"❌ 设计配置检查失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_ports(self):
        """获取所有端口名称
        
        返回: 端口名称列表
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            ports = []
            try:
                # 新方法: 使用 excitation_names 避免 deprecation
                ports = self.hfss.excitation_names
                print(f"✅ 使用 excitation_names 获取端口: {ports}")
            except AttributeError:
                try:
                    # 备用: get_excitations()
                    ports = self.hfss.get_excitations()
                    print(f"✅ 使用 get_excitations 获取端口: {ports}")
                except Exception as exc:
                    print(f"⚠️ 备用方法失败: {exc}")
            
            # 如果空，尝试常见名称
            if not ports:
                port_candidates = ["1", "Port1", "1:1", "Port_1:1"]
                for candidate in port_candidates:
                    try:
                        if hasattr(self.hfss, 'excitation_names') and candidate in self.hfss.excitation_names:
                            ports = [candidate]
                            print(f"✅ Fallback 端口: {ports}")
                            break
                    except Exception:
                        continue
            
            if not ports:
                ports = ["1:1"]  # 默认 lumped port
                print("⚠️ 使用默认端口 '1:1'")
            
            print(f"✅ 最终端口列表: {ports}")
            return ports
        except Exception as e:
            print(f"❌ 获取端口失败: {str(e)}")
            return ["1:1"]
    
    def set_variable(self, variable_name, value, unit=None):
        """
        设置变量值（支持标量和数组，带单位支持）
        
        :param variable_name: 变量名称
        :param value: 新值（标量如 5，或数组如 [2,1,1,...] 或 np.array([2,1,1,...])）
        :param unit: 单位 (如 "mm", "deg", "GHz"等)，对于数组会应用于每个元素
        返回: True 设置成功, False 设置失败
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            # 智能推断单位类型（如果未指定）
            var_lower = variable_name.lower()
            if unit is None:
                if any(kw in var_lower for kw in ["length", "width", "height", "radius", "thick"]):
                    unit = self.model_units if self.model_units else self.default_length_unit
                elif any(kw in var_lower for kw in ["angle", "theta", "phi"]):
                    unit = self.default_angle_unit
                else:
                    unit = ""  # 无量纲量
            
            # 处理 value：标量或数组
            if isinstance(value, (list, np.ndarray)):
                # 转换为 list（如果是从 np.ndarray）
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                
                # 为每个元素添加单位（如果有）
                if unit:
                    value_parts = [f"{v}{unit}" for v in value]
                else:
                    value_parts = [str(v) for v in value]
                
                # 组合成 HFSS 数组字符串：[elem1,elem2,...]
                value_str = "[" + ",".join(value_parts) + "]"
            else:
                # 标量处理（原逻辑）
                value_str = f"{value}{unit}" if unit else str(value)
            
            # 使用变量管理器设置变量
            self.hfss.variable_manager[variable_name] = value_str
            print(f"✅ 设置变量 {variable_name} = {value_str}")
            return True
        except Exception as e:
            print(f"❌ 设置变量失败: {str(e)}")
            return False
    
    def analyze(self):
        """运行仿真
        
        返回: True 仿真成功, False 仿真失败
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            print(f"\n🚀 启动仿真: {self.setup_name}...")
            start_time = time.time()
            
            # 执行仿真
            self.hfss.analyze_setup(self.setup_name)
            
            # 验证解决方案（修复: 检查solved状态，而非validate）
            print("🔍 验证解决方案...")
            setup = self.hfss.get_setup(self.setup_name)
            if setup and hasattr(setup, 'is_solved'):
                if setup.is_solved:
                    print("✅ 解决方案已解决")
                else:
                    print("⚠️ 解决方案未完全解决 - 检查HFSS日志")
            else:
                print("⚠️ 无法检查解决方案状态")
            
            elapsed = time.time() - start_time
            print(f"✅ 仿真完成! 耗时: {elapsed:.2f}秒")
            return True
        except Exception as e:
            print(f"❌ 仿真失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_s_params(self, port_combinations=None, batch_size=1, data_format="both"):
        """
        获取 S 参数结果 (优化版: 修复get_solution_data调用、频率fallback和端口标准化)
        
        :param port_combinations: 端口组合列表，如 [('1:1','1:1')] 或 [('1','1')]
        :param batch_size: 此参数保留但不再使用（为了接口兼容）
        :param data_format: 数据格式 ("dB" - 仅dB格式, "complex" - 仅复数格式, "both" - 两者都获取)
        返回: 包含所有 S 参数的 DataFrame
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            # 确定扫频路径
            sweep_path = f"{self.setup_name} : {self.sweep_name}" if self.sweep_name else self.setup_name
            print(f"🔍 获取 S 参数矩阵 (扫频路径: {sweep_path})")
            
            # 获取所有端口
            ports = self.get_ports()
            port_names = sorted(ports)  # 确保顺序一致
            
            # 如果没有指定端口组合，生成所有可能的组合
            if port_combinations is None:
                port_combinations = [(p1, p2) for p1 in port_names for p2 in port_names]
            
            # 创建结果 DataFrame
            result_df = pd.DataFrame()
            
            # 创建报告对象
            print("📈 创建标准报告...")
            report = self.hfss.post.reports_by_category.standard(setup=sweep_path)
            if not report:
                print("❌ 无法创建报告对象 - 检查sweep_path")
                return None
            
            # 设置报告属性
            report.domain = "Sweep"  # 或 "Freq" 如果是频率域
            print(f"✅ 报告域设置为: {report.domain}")
            
            # 设置报告表达式（动态标准化端口: '1' → '1:1' if needed）
            expressions = []
            for tx, rx in port_combinations:
                # 标准化端口（常见lumped port格式）
                tx_clean = tx.replace(" ", "").replace("1", "1:1") if "1" in tx and ":" not in tx else tx.replace(" ", "")
                rx_clean = rx.replace(" ", "").replace("1", "1:1") if "1" in rx and ":" not in rx else rx.replace(" ", "")
                complex_expr = f"S({tx_clean},{rx_clean})"
                db_expr = f"dB(S({tx_clean},{rx_clean}))"
                
                if data_format in ["dB"]:
                    expressions.append(db_expr)
                elif data_format in ["complex"]:
                    expressions.append(complex_expr)
                else:
                    expressions.append(db_expr)
                    expressions.append(complex_expr)
            
            report.expressions = expressions
            print(f"✅ 表达式设置: {expressions}")
            
            # 正式创建报告（关键）
            print("📊 创建报告...")
            report.create()  # 确保报告生成
            print("✅ 报告创建成功")
            
            # 获取频率点数组（修复fallback）
            frequencies = None
            try:
                if self.sweep_name:
                    sweep = self.hfss.setups[self.setup_name].sweeps[self.sweep_name]
                    if hasattr(sweep, 'solution_frequencies'):
                        frequencies = np.array(sweep.solution_frequencies) * 1e9  # Hz
                    elif hasattr(sweep, 'frequencies'):
                        frequencies = np.array(sweep.frequencies) * 1e9
                else:
                    setup = self.hfss.get_setup(self.setup_name)
                    if hasattr(setup, 'solution_frequencies'):
                        frequencies = np.array(setup.solution_frequencies) * 1e9
                    elif hasattr(setup, 'frequencies'):
                        frequencies = np.array(setup.frequencies) * 1e9
                
                print(f"✅ 从setup获取频率: {len(frequencies) if frequencies is not None else 0} 点")
            except Exception as freq_err:
                print(f"⚠️ 频率获取失败: {freq_err}，尝试报告fallback")
                # Fallback: 从报告获取（无参数调用）
                try:
                    temp_data = report.get_solution_data()  # 修复: 无参数
                    if temp_data and hasattr(temp_data, 'primary_sweep_values'):
                        frequencies = np.array(temp_data.primary_sweep_values)
                        print(f"✅ Fallback频率: {len(frequencies)} 点")
                    else:
                        frequencies = np.linspace(1e9, 3e9, 50)  # 默认1-3GHz采样
                        print("⚠️ 使用默认频率采样")
                except Exception as fb_err:
                    print(f"❌ Fallback失败: {fb_err}")
                    frequencies = np.linspace(1e9, 3e9, 50)
            
            if frequencies is None or len(frequencies) == 0:
                print("❌ 频率为空，返回None")
                return None
            
            # 获取报告数据（修复: 无参数 + 重试）
            print("📈 获取解决方案数据...")
            report_data = None
            for retry in range(3):  # 重试3次
                try:
                    report_data = report.get_solution_data()  # 修复: 无参数
                    if report_data is not None:
                        break
                except Exception as gd_err:
                    print(f"⚠️ get_solution_data尝试{retry+1}/3失败: {gd_err}")
                print(f"⚠️ 尝试{retry+1}/3: 数据加载失败，等待2s重试...")
                time.sleep(2)
            
            if report_data is None:
                print("❌ 多次尝试后仍无法获取报告数据 - 检查analyze是否完成或setup solved")
                # 调试: 检查解决方案状态
                setup = self.hfss.get_setup(self.setup_name)
                if setup and hasattr(setup, 'is_solved'):
                    print(f"调试: Setup状态 - Solved: {setup.is_solved}")
                return None
            
            # 添加频率到DataFrame
            result_df["Frequency"] = frequencies
            
            # 处理每个表达式（添加错误处理）
            for expr in expressions:
                try:
                    if 'dB' in expr:
                        # dB是实数
                        data = report_data.data_real(expr)
                        if data is not None and len(data) > 0:
                            result_df[expr] = data
                            print(f"✅ dB数据: {expr} ({len(data)}点)")
                        else:
                            print(f"⚠️ dB数据为空 for {expr}")
                    else:
                        # 复数: 实部 + 虚部
                        real_part = report_data.data_real(expr)
                        imag_part = report_data.data_imag(expr)
                        if real_part is not None and imag_part is not None and len(real_part) > 0:
                            complex_data = [complex(r, i) for r, i in zip(real_part, imag_part)]
                            result_df[expr] = complex_data
                            print(f"✅ 复数数据: {expr} ({len(complex_data)}点)")
                        else:
                            print(f"⚠️ 复数数据缺失 for {expr}")
                except Exception as expr_err:
                    print(f"⚠️ 表达式 {expr} 处理失败: {expr_err}")
            
            # 数据预览（保持原样）
            if not result_df.empty:
                print("\n📊 S 参数数据预览:")
                print(result_df.head(3))
                print(f"  数据点数: {len(result_df)}")
                print(f"  参数数量: {len(result_df.columns) - 1}")
                
                # 复数验证（保持原样）
                complex_cols = [col for col in result_df.columns if col.startswith('S(') and 'dB' not in col]
                if complex_cols:
                    print("\n复数S参数验证:")
                    for col in complex_cols:
                        sample = result_df[col].iloc[0]
                        if isinstance(sample, complex):
                            print(f"  {col}: complex 示例: {sample}")
                        elif isinstance(sample, float):
                            print(f"  {col}: float 示例: {sample}")
                        else:
                            print(f"  {col}: 未知类型 {type(sample)}")
                else:
                    print("⚠️ 未检测到复数格式S参数数据")
            else:
                print("❌ 未获取到有效数据")
            
            return result_df

        except Exception as e:
            print(f"❌ 获取 S 参数失败: {str(e)}")
            traceback.print_exc()
            return None

    def get_farfield_data(self, sphere_name="3D", frequencies=None, quantity="GainTotal", data_format="dB"):
        """
        获取远场数据（如 GainTotal in dB）
        
        :param sphere_name: 远场球体名称 (默认: "3D")
        :param frequencies: 频率列表 (Hz)，如 [10e9] 或 None (使用所有频率)
        :param quantity: 远场量 (默认: "GainTotal"，其他如 "Directivity")
        :param data_format: 数据格式 ("dB" - dB 格式, "mag" - 幅度)
        返回: 包含远场数据的 DataFrame (列: Frequency, Theta, Phi, {quantity}_{data_format})
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            # 确定扫频路径
            sweep_path = f"{self.setup_name} : {self.sweep_name}" if self.sweep_name else self.setup_name
            
            print(f"🔍 获取远场数据: {quantity} ({data_format}), 球体: {sphere_name}, 扫频路径: {sweep_path}")
            
            # 构建表达式
            if data_format == "dB":
                expr = f"dB({quantity})"
            elif data_format == "mag":
                expr = f"Mag({quantity})"
            else:
                expr = quantity
            expressions = [expr]
            
            # 准备频率变异（**variations）
            variations = {}
            if frequencies:
                freq_ghz_str = [f"{f / 1e9}GHz" for f in frequencies]
                variations["Freq"] = freq_ghz_str[0] if len(freq_ghz_str) == 1 else freq_ghz_str
                print(f"  设置频率变异: Freq={variations['Freq']}")
            
            # 步骤1: 创建远场报告（传入 **variations）
            print("  步骤1: 创建报告对象...")
            report = self.hfss.post.reports_by_category.far_field(
                expressions=expressions,
                setup=sweep_path,
                sphere_name=sphere_name,
                **variations  # 关键：在这里设置频率
            )
            if not report:
                print("❌ 步骤1失败: 无法创建远场报告对象")
                return None
            print("  ✅ 步骤1成功: 报告对象创建")
            
            # 步骤2: 设置扫频属性
            print("  步骤2: 设置扫频...")
            report.primary_sweep = "Phi"    # 主扫频: Phi (0-360°)
            report.secondary_sweep = "Theta" # 副扫频: Theta (0-180°)
            print("  ✅ 步骤2成功: 扫频设置")
            
            # 步骤3: 设置域并创建报告
            print("  步骤3: 设置域并创建报告...")
            report.domain = "Sweep"
            report.create()  # 关键：正式创建报告
            print("  ✅ 步骤3成功: 报告已创建")
            
            # 步骤4: 获取解决方案数据
            print("  步骤4: 获取解决方案数据...")
            solution_data = report.get_solution_data()
            if solution_data is None:
                print("❌ 步骤4失败: 无法获取解决方案数据")
                return None
            print("  ✅ 步骤4成功: 解决方案数据获取")
            
            # 步骤5: 获取扫频值（使用 variation_values 方法）
            print("  步骤5: 获取变异扫频值...")
            try:
                phi_values = solution_data.variation_values("Phi")
                theta_values = solution_data.variation_values("Theta")
                print(f"  Phi 值: {len(phi_values)} 点 ({phi_values.min():.1f}~{phi_values.max():.1f}°)")
                print(f"  Theta 值: {len(theta_values)} 点 ({theta_values.min():.1f}~{theta_values.max():.1f}°)")
            except Exception as ve:
                print(f"⚠️ 变异获取失败 ({ve})，使用默认范围")
                # Fallback: 标准远场网格
                phi_values = np.arange(0, 360.1, 5)  # 0-360° step 5°
                theta_values = np.arange(0, 180.1, 5)  # 0-180° step 5°
                print(f"  Fallback Phi: {len(phi_values)} 点 (0~360°)")
                print(f"  Fallback Theta: {len(theta_values)} 点 (0~180°)")
            
            # 处理频率（尝试变异，fallback 到指定或 setup）
            freq_values = None
            try:
                freq_values = solution_data.variation_values("Freq")
                if freq_values is not None:
                    freq_values = np.array(freq_values) * 1e9  # GHz -> Hz
            except:
                pass
            if freq_values is None:
                if frequencies:
                    freq_values = np.array(frequencies)
                else:
                    # 从 setup 获取
                    setup = self.hfss.get_setup(self.setup_name)
                    if setup and hasattr(setup, 'solution_frequencies'):
                        freq_values = np.array(setup.solution_frequencies) * 1e9
                    else:
                        freq_values = np.array([5e9])  # 默认
            print(f"  Freq 值: {len(freq_values)} 点 ({freq_values.min()/1e9:.2f}~{freq_values.max()/1e9:.2f}GHz)")
            
            # 步骤6: 获取数据（优先 data_real for dB/mag）
            print("  步骤6: 获取表达式数据...")
            data_array = solution_data.data_real(expressions[0])  # 实部 (dB 是实数)
            if data_array is None:
                # 备选
                data_array = solution_data.get_expression_data(expressions[0])
                if data_array is None:
                    print(f"❌ 步骤6失败: 无法获取 {expressions[0]} 数据")
                    return None
            print(f"  ✅ 步骤6成功: 数据形状 {np.shape(data_array)}")
            
            # 步骤7: 展平数据网格（支持单/多频）
            print("  步骤7: 展平数据网格...")
            if len(freq_values) == 1:
                # 单频: Theta x Phi 网格 (indexing='ij' 确保 [n_theta, n_phi])
                Theta_grid, Phi_grid = np.meshgrid(theta_values, phi_values, indexing='ij')
                flat_theta = Theta_grid.flatten()
                flat_phi = Phi_grid.flatten()
                flat_data = np.array(data_array).flatten()  # 假设 data_array 匹配 [n_theta, n_phi]
                if len(flat_data) != len(flat_theta):
                    print(f"⚠️ 数据形状不匹配 ({len(flat_data)} vs {len(flat_theta)})，调整展平")
                    flat_data = np.resize(flat_data, len(flat_theta))  # 简单调整
                frequencies_flat = np.full(len(flat_theta), freq_values[0])
            else:
                # 多频: Freq x Theta x Phi -> flatten
                n_theta, n_phi = len(theta_values), len(phi_values)
                Theta_grid, Phi_grid = np.meshgrid(theta_values, phi_values, indexing='ij')
                flat_theta_base = Theta_grid.flatten()
                flat_phi_base = Phi_grid.flatten()
                flat_theta = np.tile(flat_theta_base, len(freq_values))
                flat_phi = np.tile(flat_phi_base, len(freq_values))
                # 假设 data_array 是 [n_freq, n_theta, n_phi] 或需 reshape
                if len(np.shape(data_array)) == 3:
                    flat_data = data_array.reshape(-1)
                else:
                    # Fallback: 重复单频数据
                    flat_data_base = np.array(data_array).flatten()
                    flat_data = np.tile(flat_data_base, len(freq_values))
                frequencies_flat = np.repeat(freq_values, n_theta * n_phi)
            flat_data = [float(val) for val in flat_data]
            print(f"  ✅ 步骤7成功: 展平完成，数据点数={len(flat_data)}")
            
            # 步骤8: 构建 DataFrame
            result_df = pd.DataFrame({
                'Frequency': frequencies_flat,
                'Theta': flat_theta,
                'Phi': flat_phi,
                f'{quantity}_{data_format}': flat_data
            })
            
            # 数据预览
            print("\n📊 远场数据预览:")
            print(result_df.head(5))
            print(f"  数据点数: {len(result_df)}")
            if not result_df.empty:
                print(f"  示例值 (Theta={flat_theta[0]:.1f}°, Phi={flat_phi[0]:.1f}°): {result_df[f'{quantity}_{data_format}'].iloc[0]:.2f}")
            
            return result_df

        except Exception as e:
            print(f"❌ 获取远场数据失败: {str(e)}")
            traceback.print_exc()
            return None

    def save_s_params(self, s_params, output_csv=None):
        """保存原始S参数数据到CSV文件"""
        if output_csv is None:
            import tempfile
            output_csv = os.path.join(
                tempfile.gettempdir(),
                f"{os.path.basename(self.project_path).replace('.aedt', '')}_s_params.csv"
            )
        
        try:
            # 处理相对路径
            if not os.path.isabs(output_csv):
                output_csv = os.path.abspath(output_csv)
            
            # 确保目录存在
            dir_path = os.path.dirname(output_csv)
            if dir_path:  # 如果有目录路径
                os.makedirs(dir_path, exist_ok=True)
            else:
                # 如果没有目录，设置为当前工作目录
                output_csv = os.path.join(os.getcwd(), os.path.basename(output_csv))
            
            # 保存为CSV
            s_params.to_csv(output_csv, index=False)
            print(f"💾💾 原始S参数已保存至: {output_csv}")
            return output_csv
        except Exception as e:
            print(f"❌❌ 保存S参数失败: {str(e)}")
            return None

    def save_project(self, new_path=None):
        """保存项目

        :param new_path: 可选的新路径
        返回: True 保存成功, False 保存失败
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            if new_path:
                self.hfss.save_project(new_path)
                print(f"💾 项目已另存为: {new_path}")
            else:
                self.hfss.save_project()
                print("💾 项目已保存")
            return True
        except Exception as e:
            print(f"❌ 保存失败: {str(e)}")
            return False

    def close(self):
        """关闭 HFSS 连接

        返回: True 关闭成功, False 关闭失败
        """
        try:
            # 先释放matplotlib资源
            import matplotlib.pyplot as plt
            plt.close('all')
            # 再关闭HFSS连接
            if self.hfss:
                print("🛑 正在关闭 ANSYS...")
                self.hfss.close_desktop()
                print("✅ ANSYS 已关闭")
                self.hfss = None
                self._desktop = None
                # 添加延迟确保资源释放
                time.sleep(5)
            return True
    
        except Exception as e:
            print(f"❌ 关闭失败: {str(e)}")
            return False

    def export_results(self, df, output_csv=None, max_retries=3):
        """导出结果到CSV文件"""
        try:
            if output_csv is None:
                import tempfile
                output_csv = os.path.join(
                    tempfile.gettempdir(),
                    os.path.basename(self.project_path).replace(".aedt", "_results.csv")
                )
            
            # 确保输出路径是文件而非目录
            if os.path.isdir(output_csv):
                output_csv = os.path.join(output_csv, "hfss_results.csv")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            
            for i in range(max_retries):
                try:
                    df.to_csv(output_csv, index=False)
                    print(f"💾 结果已导出至: {output_csv}")
                    return output_csv
                except PermissionError as pe:
                    if i < max_retries - 1:
                        print(f"⚠️ 文件占用中，等待重试 ({i+1}/{max_retries})...")
                        time.sleep(30)  #等待30秒
                    else:
                        print(f"❌ 多次尝试失败: {str(pe)}")
                        return None
        except Exception as e:
            print(f"❌ 导出结果失败: {str(e)}")
            return None

    def __enter__(self):
        """上下文管理器入口 - 自动连接"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """上下文管理器出口 - 自动关闭"""
        self.close()