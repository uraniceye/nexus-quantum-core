# --- START OF FILE quantum_core.py ---

# -*- coding: utf-8 -*-
"""
NEXUS QUANTUM DEFENSE - 量子计算核心库 v1.5.0 (Topology-Aware API Version)

功能:
- 封装所有量子态模拟 (QuantumState)、量子线路 (QuantumCircuit) 的核心数据结构与逻辑。
- 采用惰性求值与双模式（态矢量/密度矩阵）模拟引擎，兼顾纯态效率与混合态功能。
- 包含高性能的并行计算基础设施，用于加速大规模量子模拟。
- 提供一个简洁、稳定、健 robuste的公共API。

增强功能 (v1.5.0):
- API 升级: 公共 API `run_circuit_on_state` 和相关内部方法现在接收 `topology` 参数，
            以实现对量子子程序的拓扑感知优化。
- 架构重构: 彻底移除了 get_effective_unitary 中的猴子补丁，采用内置的模式感知执行。
- 噪声模型扩展: 引入关联噪声模型 (CorrelatedNoise)，模拟串扰等非局部效应。
- 逻辑修复: 修正了相干噪声应用逻辑，确保其只作用于旋转角度。
- 优化内核调度: _StateVectorEntity 内部使用调度字典，提高代码可维护性。

Author: 跳舞的火公子
Date: 2025-10-28
"""

# ========================================================================
# --- 1. 导入依赖 ---
# ========================================================================

# --- 1.1 Python标准库 ---
import sys
import os
import logging
import copy
import math
import cmath
import uuid
import time
import types
import random
from typing import Optional, List, Tuple, Dict, Any, Literal, Union, ClassVar, Callable
from dataclasses import dataclass, field, fields # 确保 dataclass, field, fields 已导入
from datetime import datetime, timezone
from functools import lru_cache
from abc import ABC, abstractmethod

# 纯Python并行计算框架所需
import multiprocessing as mp
from multiprocessing import pool, RawArray
import ctypes


# --- 1.2 可选的高性能GPU依赖 ---
try:
    import cupy as cp
except ImportError:
    cp = None
    logging.getLogger(__name__).info(
        "CuPy library not found. GPU acceleration will be unavailable. "
        "The simulation will use the Pure Python backend by default."
    )

# --- 1.3 日志记录器初始化 ---
logger = logging.getLogger(__name__)


# ========================================================================
# --- 2. 硬件资源配置 ---
# ========================================================================

# --- 2.1 硬件资源常量 ---
# 这些常量与具体后端无关，因此保持不变
_BYTES_PER_COMPLEX128_ELEMENT: int = 16
_GB_TO_BYTES: int = 1024 ** 3
_TOTAL_SYSTEM_RAM_GB_HW: int = 128  # 示例硬件配置 (现在作为最终的回退值)
_TOTAL_GPU_VRAM_GB_HW: int = 24   # 示例硬件配置 (GPU VRAM 无法动态获取)


# --- [新增] 动态内存检测的辅助函数与缓存 ---
_LAST_RAM_CHECK_TIME = 0
_CACHED_AVAILABLE_RAM_GB = 0

def _get_available_system_ram_gb() -> Optional[float]:
    """
    [硬编码实现] 尝试在不使用外部库的情况下，动态获取系统可用物理内存。

    此函数会根据操作系统尝试不同的方法：
    -   Windows: 使用 ctypes 调用 Kernel32.dll 的 GlobalMemoryStatusEx 函数。
    -   Linux: 读取 /proc/meminfo 文件并解析 'MemAvailable' 字段。
    -   macOS: 尝试使用 vm_stat 系统命令并解析输出。

    如果所有方法都失败，它会返回 None，表示应使用硬编码的备用值。
    为了避免频繁调用系统API，此函数会对结果进行简单的缓存（10秒）。

    Returns:
        Optional[float]: 可用物理内存的GB数，如果无法获取则返回 None。
    """
    global _LAST_RAM_CHECK_TIME, _CACHED_AVAILABLE_RAM_GB
    
    # 简单的10秒缓存机制
    current_time = time.time()
    if current_time - _LAST_RAM_CHECK_TIME < 10 and _CACHED_AVAILABLE_RAM_GB > 0:
        return _CACHED_AVAILABLE_RAM_GB

    available_ram_gb = None

    try:
        if sys.platform.startswith("win32"):
            # --- Windows 实现 ---
            # 使用 ctypes 直接调用 Windows API
            kernel32 = ctypes.windll.kernel32
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem_status = MEMORYSTATUSEX()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)) != 0:
                available_ram_bytes = mem_status.ullAvailPhys
                available_ram_gb = available_ram_bytes / _GB_TO_BYTES

        elif sys.platform.startswith("linux"):
            # --- Linux 实现 ---
            # 读取 /proc/meminfo 文件
            with open("/proc/meminfo", "r") as meminfo_file:
                for line in meminfo_file:
                    if "MemAvailable:" in line:
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            available_ram_kb = int(parts[1])
                            available_ram_gb = available_ram_kb / (1024 * 1024) # KB to GB
                            break

        elif sys.platform.startswith("darwin"):
            # --- macOS 实现 ---
            # 使用 vm_stat 命令，这比较脆弱，但没有外部库的情况下是可行方案之一
            import subprocess
            try:
                result = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True)
                lines = result.stdout.splitlines()
                page_size = 0
                free_pages = 0
                inactive_pages = 0
                
                # 获取页面大小
                for line in lines:
                    if line.startswith("page size of"):
                        page_size_str = line.split()[3]
                        if page_size_str.isdigit():
                            page_size = int(page_size_str)
                            break
                
                if page_size > 0:
                    for line in lines:
                        if line.startswith("Pages free:"):
                            free_pages_str = line.split()[2].strip('.')
                            if free_pages_str.isdigit():
                                free_pages = int(free_pages_str)
                        elif line.startswith("Pages inactive:"):
                            inactive_pages_str = line.split()[2].strip('.')
                            if inactive_pages_str.isdigit():
                                inactive_pages = int(inactive_pages_str)

                    # macOS 的 "可用内存" 通常被认为是 free + inactive
                    available_ram_bytes = (free_pages + inactive_pages) * page_size
                    available_ram_gb = available_ram_bytes / _GB_TO_BYTES

            except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
                # 如果 vm_stat 失败，则不执行任何操作，让 available_ram_gb 保持为 None
                logger.warning("Could not get available RAM using 'vm_stat' on macOS. Will use fallback.", exc_info=True)
                pass
                
    except Exception as e:
        # 捕获所有可能的意外错误，确保函数不会崩溃
        logger.warning(f"An unexpected error occurred during dynamic RAM check: {e}. Will use fallback.", exc_info=True)
        available_ram_gb = None
    
    if available_ram_gb is not None:
        _CACHED_AVAILABLE_RAM_GB = available_ram_gb
        _LAST_RAM_CHECK_TIME = current_time
        return available_ram_gb
    else:
        # 如果所有方法都失败了，确保缓存被清除，以便下次可以重试
        _CACHED_AVAILABLE_RAM_GB = 0
        return None

# --- 2.2 库的内部可配置参数 (终极修正版) ---
_core_config: Dict[str, Any] = {
    # 用户可设置的最大量子比特数上限
    "MAX_QUBITS": 30,
    
    # 后端选择配置。可用选项: 'auto', 'cupy', 'pure_python'
    "BACKEND_CHOICE": "auto",
    
    # 触发纯Python并行计算的量子比特数阈值。
    "PARALLEL_COMPUTATION_QUBIT_THRESHOLD": 12,
    
    # [新增] 用户可手动设置系统 RAM 上限 (GB)。如果为 None，则尝试动态检测。
    "MAX_SYSTEM_RAM_GB_LIMIT": None,
}

# [新增] 全局占位符变量，用于解决 get_effective_unitary 的循环依赖
_get_effective_unitary_placeholder: Optional[Callable] = None

def configure_quantum_core(config_dict: Dict[str, Any]):
    """
    公共API函数：用于在运行时配置量子核心库的行为。
    [终极修正版] 移除了对 NumPy 后端的支持，并增加了内存限制配置。
    """
    global _core_config
    if not isinstance(config_dict, dict):
        logger.warning(f"Failed to configure quantum core: provided config_dict is not a dictionary.")
        return

    # 准备一个列表来收集不再有效的配置键
    inactive_keys = []
    
    # 检查是否请求了已被移除的 'numpy' 后端
    if config_dict.get("BACKEND_CHOICE") == "numpy":
        logger.error(
            "Configuration Error: The 'numpy' backend is no longer supported in this version of the library. "
            "The configuration 'BACKEND_CHOICE: \"numpy\"' is invalid. "
            "Falling back to 'pure_python'."
        )
        # 强制回退到有效的后端，而不是接受一个无效的配置
        config_dict["BACKEND_CHOICE"] = "pure_python"

    # 检查是否传入了旧的、已弃用的 'USE_GPU_FOR_QUANTUM_STATE' 配置
    if "USE_GPU_FOR_QUANTUM_STATE" in config_dict:
        inactive_keys.append("USE_GPU_FOR_QUANTUM_STATE")
        logger.warning(
            "The configuration key 'USE_GPU_FOR_QUANTUM_STATE' is deprecated and has been ignored. "
            "Please use 'BACKEND_CHOICE': 'cupy' or 'auto' instead."
        )
        # 如果用户意图是使用GPU但没有设置新配置，则帮助他们转换
        if config_dict["USE_GPU_FOR_QUANTUM_STATE"] is True and "BACKEND_CHOICE" not in config_dict:
            config_dict["BACKEND_CHOICE"] = "auto"
    
    # 打印所有已知的无效/被忽略的配置键
    if inactive_keys:
        logger.warning(
            f"The following configuration keys have no effect or are deprecated: "
            f"{', '.join(inactive_keys)}"
        )
    
    # 更新全局配置字典
    valid_keys = _core_config.keys()
    for key, value in config_dict.items():
        if key in valid_keys:
            _core_config[key] = value
        elif key not in inactive_keys: # 避免重复警告
            logger.warning(f"Ignoring unknown configuration key: '{key}'. Valid keys are: {list(valid_keys)}.")

    logger.info(f"Quantum Core library has been configured with new settings: {_core_config}")


# ========================================================================
# --- 3. [核心] 基础数据结构与抽象基类 ---
# ========================================================================
@dataclass
class NoiseModel(ABC):
    """
    一个抽象基类，用于定义可应用于量子线路的可插拔噪声模型。

    任何具体的噪声模型都应继承自此类，并实现 `get_noise_for_op` 方法。
    这个设计模式允许 `QuantumState.run_circuit` 方法以一种通用的方式
    与不同的噪声模型进行交互，而无需了解其内部实现细节。

    这种架构将噪声的“定义”（由具体的 NoiseModel 子类完成）与噪声的
    “应用”（由 QuantumState 核心引擎完成）清晰地分离开来，极大地提高了
    代码的模块化、可扩展性和可维护性。
    """
    
    # 修正：移除了 dataclasses.field 定义，因为 NoiseModel 不是 dataclass
    # _internal_logger: logging.Logger = field(default_factory=lambda: logging.getLogger(f"{__name__}.{__class__.__name__}"), repr=False, init=False)

    def __init__(self):
        """
        初始化噪声模型的基类。
        每个 NoiseModel 实例都会有一个自己的日志记录器。
        """
        self._internal_logger: logging.Logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._internal_logger.debug(f"NoiseModel base class initialized for {self.__class__.__name__}.")


    @abstractmethod
    def get_noise_for_op(self, op_name: str, qubits: List[int]) -> Dict[str, Any]:
        """
        [抽象方法] 根据给定的操作名称和作用的量子比特，返回要应用的噪声。

        此方法必须由所有具体的噪声模型子类实现。它定义了噪声模型与
        量子线路执行引擎之间的核心接口。

        Args:
            op_name (str): 
                门操作的名称 (e.g., 'h', 'cnot', 'rx')。
            qubits (List[int]): 
                门操作作用的量子比特的列表。这些量子比特索引应为全局索引。

        Returns:
            Dict[str, Any]:
                一个描述噪声的字典。如果对于给定的操作没有噪声，
                则应返回一个空字典 (`{}`)。字典可以包含以下键：

                - 'coherent_error' (Optional[Dict[str, Any]]):
                  一个描述门参数误差的字典。
                  例如: `{'angle_error': 0.01}` 表示给旋转门增加 0.01 弧度的误差。
                  执行引擎将会在应用理想门操作*之前*修改其参数。

                - 'coherent_unitary_replacement' (Optional[Dict[str, Any]]):
                  一个字典，包含一个 `QuantumCircuit` 实例，用于完全替换原始的理想门。
                  例如: `{'circuit': my_noisy_subcircuit}`。
                  这使得可以模拟相干串扰或更复杂的门失真。如果此键存在，
                  则 'coherent_error' 和 'incoherent_post_op' 会被忽略。

                - 'incoherent_post_op' (Optional[List[Dict[str, Any]]]):
                  一个在理想门操作*之后*应用的量子通道（非相干噪声）列表。
                  列表中的每个字典都描述了一个 `apply_quantum_channel` 调用，
                  例如: `[{'type': 'depolarizing', 'qubits': [0, 1], 'params': {'probability': 0.001}}]`
        """
        raise NotImplementedError("Subclasses of NoiseModel must implement the get_noise_for_op method.")

    def get_readout_error_matrix(self, num_qubits: int) -> Optional[Dict[int, float]]:
        """
        [可选接口] 返回一个描述读出错误的模型。

        读出错误是经典错误，它发生在测量的最后阶段，会“污染”理想的概率分布。
        一个具体的噪声模型可以选择性地覆盖此方法。

        Args:
            num_qubits (int): 
                当前量子系统的总比特数。

        Returns:
            Optional[Dict[int, float]]:
                一个字典，将量子比特索引映射到其读出错误率 (0.0 到 1.0 之间)。
                例如: `{0: 0.01, 1: 0.015}`。
                如果模型不包含读出错误，则应返回 `None`。
        """
        # 默认实现返回 None，表示没有读出错误。
        self._internal_logger.debug("Default NoiseModel: No readout error defined, returning None.")
        return None

@dataclass
class PauliString:
    """
    [健壮性改进版 - v1.5.1 修正] 表示一个带系数的Pauli串，例如 `0.5 * 'IXYZ'`。

    这是一个纯粹的数据结构，用于在哈密顿量中表示一个单独的项。它与任何
    特定的计算后端（CuPy, or PurePythonBackend）完全解耦，确保了上层算法
    定义的通用性。

    [v1.5.1 修正]:
    - 修正了 `__eq__` 和 `__hash__` 方法中数值比较逻辑不一致的潜在错误。
      现在，两个方法都使用统一的、基于固定精度的舍入方法来处理复数系数，
      从而严格遵守 Python 数据模型中 `a == b` 必须保证 `hash(a) == hash(b)`
      的核心原则。这消除了在将 PauliString 对象用作字典键或存入集合时
      可能出现的罕见但严重的哈希冲突或查找失败问题。

    Attributes:
        coefficient (complex):
            Pauli串的复数系数。在初始化时，整数或浮点数会被自动转换
            为复数类型，以保证数值运算的一致性。
        pauli_map (Dict[int, Literal['I', 'X', 'Y', 'Z']]):
            一个字典，将量子比特的索引（非负整数）映射到作用于该比特的
            Pauli算子类型（'I', 'X', 'Y', 'Z'）。
            如果一个量子比特的索引没有出现在这个字典中，则默认其上的操作
            是单位算子 'I'。

    Example:
        # 表示 0.5 * X_0 Z_2 I_1 (在一个至少3比特的系统上)
        ps1 = PauliString(coefficient=0.5, pauli_map={0: 'X', 2: 'Z'})

        # 表示 (1+2j) * Y_1
        ps2 = PauliString(coefficient=1+2j, pauli_map={1: 'Y'})

        # 表示 -1.0 * I (纯数乘)
        ps3 = PauliString(coefficient=-1.0, pauli_map={})
    """
    coefficient: complex
    pauli_map: Dict[int, Literal['I', 'X', 'Y', 'Z']] = field(default_factory=dict)

    # [核心修正] 定义一个类级别的常量，用于统一舍入精度
    _HASH_PRECISION_DIGITS: ClassVar[int] = 9

    def __post_init__(self):
        """
        在对象初始化后自动调用的验证与规范化方法。

        此方法由 `dataclasses` 模块自动调用。它的核心作用是确保每个
        `PauliString` 实例在创建时都处于一个合法的、一致的状态，从而
        防止无效数据在系统中传播。

        执行的检查包括：
        1.  `coefficient` 必须是数值类型 (complex, float, or int)，并被规范化为 `complex`。
        2.  `pauli_map` 必须是一个字典。
        3.  `pauli_map` 的键 (qubit_index) 必须是无符号整数。
        4.  `pauli_map` 的值 (operator) 必须是 'I', 'X', 'Y', 'Z' 之一。
        5.  如果 `pauli_map` 中包含 `'I'` 操作，会将其移除，因为它们是隐式的。

        Raises:
            TypeError: 如果 `coefficient` 或 `pauli_map` 的类型不正确。
            ValueError: 如果 `pauli_map` 中的量子比特索引为负数，或者算子
                        字符串无效。
        """
        # --- 步骤 1: 验证并规范化系数的类型 ---
        if not isinstance(self.coefficient, (complex, float, int)):
            raise TypeError(
                f"PauliString: 'coefficient' must be a numeric type (complex, float, or int), "
                f"but got {type(self.coefficient).__name__}."
            )
        self.coefficient = complex(self.coefficient)

        # --- 步骤 2: 确保 pauli_map 是一个字典 ---
        if not isinstance(self.pauli_map, dict):
            raise TypeError(
                f"PauliString: 'pauli_map' must be a dictionary, "
                f"but got {type(self.pauli_map).__name__}."
            )
        
        cleaned_pauli_map: Dict[int, Literal['X', 'Y', 'Z']] = {}

        # --- 步骤 3: 遍历并验证 pauli_map 的内容（键和值） ---
        for qubit_index, operator in self.pauli_map.items():
            if not isinstance(qubit_index, int) or qubit_index < 0:
                raise ValueError(
                    f"PauliString: Qubit index in 'pauli_map' must be a non-negative integer, "
                    f"but got {qubit_index} for operator '{operator}'."
                )
            
            if operator not in ['I', 'X', 'Y', 'Z']:
                raise ValueError(
                    f"PauliString: Pauli operator in 'pauli_map' must be 'I', 'X', 'Y', or 'Z', "
                    f"but got '{operator}' for qubit {qubit_index}."
                )
            
            # 移除单位算子 'I'，因为它们是隐式的
            if operator != 'I':
                cleaned_pauli_map[qubit_index] = operator
        
        # --- 步骤 4: 替换原始 pauli_map 为经过验证和清理的版本 ---
        self.pauli_map = cleaned_pauli_map

    @staticmethod
    def _round_complex(c: complex, digits: int) -> Tuple[float, float]:
        """
        [新增静态辅助方法] 以一种确定性的方式舍入一个复数。
        返回一个包含舍入后实部和虚部的元组，该元组是可哈希的。
        """
        return round(c.real, digits), round(c.imag, digits)

    def __str__(self) -> str:
        """
        提供 `PauliString` 的一个人类可读的、确定性的字符串表示。
        
        示例输出:
        - `0.5 * X0Z2`
        - `(1+2j) * Y1`
        - `-1.0 * I`

        Returns:
            str: 格式化的、代表该 PauliString 的字符串。
        """
        pauli_string_parts: List[str] = []
        
        # 按量子比特索引排序，以确保确定性输出
        for qubit_index in sorted(self.pauli_map.keys()):
            operator = self.pauli_map[qubit_index]
            pauli_string_parts.append(f"{operator}{qubit_index}")
        
        pauli_term_string = "".join(pauli_string_parts) if pauli_string_parts else "I"
        
        # 格式化系数以提高可读性
        if self.coefficient.imag == 0:
            coeff_val_real = self.coefficient.real
            if coeff_val_real == int(coeff_val_real):
                coeff_str = str(int(coeff_val_real))
            else:
                coeff_str = str(coeff_val_real)
        else:
            coeff_str = str(self.coefficient)

        return f"{coeff_str} * {pauli_term_string}"

    def __eq__(self, other: Any) -> bool:
        """
        [核心修正] 定义 PauliString 对象的相等性比较。
        
        两个 PauliString 对象相等当且仅当它们的系数（在固定精度下舍入后）
        和 Pauli Map 完全相同。此方法现在使用与 `__hash__` 方法完全一致的
        舍入逻辑，以确保 `a == b` => `hash(a) == hash(b)`。
        """
        if not isinstance(other, PauliString):
            return NotImplemented
        
        # 使用与 __hash__ 完全相同的舍入逻辑来比较系数
        coeffs_are_equal = (
            self._round_complex(self.coefficient, self._HASH_PRECISION_DIGITS) ==
            self._round_complex(other.coefficient, self._HASH_PRECISION_DIGITS)
        )
        
        pauli_maps_are_equal = self.pauli_map == other.pauli_map
        
        return coeffs_are_equal and pauli_maps_are_equal

    def __hash__(self) -> int:
        """
        [核心修正] 定义 PauliString 对象的哈希值，使其可以在集合或字典中使用。
        
        哈希值基于其规范化的、经过固定精度舍入的系数和排序后的 Pauli Map，
        以确保哈希的稳定性和与 `__eq__` 方法的一致性。
        """
        # 使用统一的舍入辅助方法来处理复数系数
        coeff_hash_part = hash(self._round_complex(self.coefficient, self._HASH_PRECISION_DIGITS))
        
        # 对 Pauli Map 进行哈希。由于字典本身不可哈希，需要将其转换为可哈希的元组。
        # 键值对必须先排序，以确保确定性哈希。
        pauli_map_hash_part = hash(tuple(sorted(self.pauli_map.items())))
        
        # 将两个部分的哈希值组合成最终的哈希
        return hash((coeff_hash_part, pauli_map_hash_part))
# --- 使用类型别名明确定义 Hamiltonian ---
Hamiltonian = List[PauliString]
"""表示哈密顿量，它是一个 PauliString 对象的列表。"""



# [新增] 定义一个临时的占位符类，以解决循环依赖问题
class _TemporaryQuantumCircuitPlaceholder:
    """
    [健壮性改进版] 一个临时的占位符类，用于在定义 `QuantumCircuit` 之前
    解决循环类型提示问题，并供 `AlgorithmBuilders` 内部构建函数使用。

    此类的实例模仿 `QuantumCircuit` 的所有公共门 API，但其唯一职责是：
    **收集原始指令元组到 `self.instructions` 列表中，而不触发任何宏查找、
    宏展开、或对指令参数的深度验证。**

    它不进行任何昂贵的计算，不维护状态缓存，也不与后端交互。
    它的存在是为了在 `AlgorithmBuilders` 内部构建“宏”时，
    能够有一个像 `QuantumCircuit` 一样调用的接口，而不会因为 `QuantumCircuit`
    尚未完全定义或因宏展开而产生循环依赖。

    Attributes:
        num_qubits (int):
            此电路所设计的量子比特数量。
        description (Optional[str]):
            此占位符电路的人类可读描述。
        instructions (List[Tuple[Any, ...]]):
            收集到的原始指令元组的序列。

    Raises:
        ValueError: 如果 `num_qubits` 为负数。
        TypeError: 如果 `instructions` 不是列表。
    """
    def __init__(self, num_qubits: int, description: Optional[str] = None, instructions: Optional[List[Tuple[Any, ...]]] = None):
        """
        初始化 _TemporaryQuantumCircuitPlaceholder 实例。

        Args:
            num_qubits (int): 
                此占位符电路所设计的量子比特数量。
            description (Optional[str], optional): 
                此占位符电路的人类可读描述。默认为 None。
            instructions (Optional[List[Tuple[Any, ...]]], optional): 
                初始指令列表。默认为 None，将初始化为空列表。
        """
        # --- 1. 输入验证 ---
        if not isinstance(num_qubits, int) or num_qubits < 0:
            raise ValueError(
                f"_TemporaryQuantumCircuitPlaceholder: 'num_qubits' must be a non-negative integer, "
                f"but got {num_qubits}."
            )
        if instructions is not None and not isinstance(instructions, list):
            raise TypeError(
                f"_TemporaryQuantumCircuitPlaceholder: 'instructions' must be a list or None, "
                f"but got {type(instructions).__name__}."
            )
        if description is not None and not isinstance(description, str):
            raise TypeError(
                f"_TemporaryQuantumCircuitPlaceholder: 'description' must be a string or None, "
                f"but got {type(description).__name__}."
            )

        # --- 2. 属性赋值 ---
        self.num_qubits = num_qubits
        self.description = description
        self.instructions = instructions if instructions is not None else []
        
        # 内部日志器，仅用于调试此占位符类
        self._internal_logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        self._internal_logger.debug(f"Placeholder circuit for {self.num_qubits} qubits initialized.")

    def _add_raw_gate(self, gate_name: str, *args: Any, **kwargs: Any):
        """
        [内部方法] 直接向占位符电路中添加一个原始门指令。
        此方法不会执行任何宏查找或展开，仅用于指令收集。
        """
        # [健壮性改进] 对 gate_name 进行基本验证
        if not isinstance(gate_name, str) or not gate_name.strip():
            self._internal_logger.error(f"Placeholder: Attempted to add gate with invalid name '{gate_name}'. Ignoring.")
            return

        instruction_list: List[Any] = [gate_name] + list(args)
        if kwargs:
            instruction_list.append(kwargs)
        self.instructions.append(tuple(instruction_list))
        self._internal_logger.debug(f"Placeholder: Added raw instruction to buffer: {tuple(instruction_list)}")

    def add_gate(self, gate_name: str, *args: Any, condition: Optional[Tuple[int, int]] = None, **kwargs: Any):
        """
        [核心方法] 向占位符电路中添加一个门操作指令。
        此方法是所有门API的统一入口。它不进行宏查找和展开，
        仅将指令以标准格式（包含kwargs和condition）添加到内部指令列表中。
        """
        # [健壮性改进] 对 gate_name 进行基本验证
        if not isinstance(gate_name, str) or not gate_name.strip():
            self._internal_logger.error(f"Placeholder: Attempted to add gate with invalid name '{gate_name}'. Ignoring.")
            return

        instruction_kwargs = dict(kwargs) 
        if condition is not None:
            # [健壮性改进] 对 condition 参数进行基本验证
            if not (isinstance(condition, tuple) and len(condition) == 2 and
                    isinstance(condition[0], int) and isinstance(condition[1], int)):
                self._internal_logger.warning(
                    f"Placeholder: Invalid condition format ('{condition}') for gate '{gate_name}'. "
                    "Condition will be stored as is, but may cause errors in later processing."
                )
            instruction_kwargs['condition'] = condition
            
        instruction_list: List[Any] = [gate_name] + list(args)
        if instruction_kwargs:
            instruction_list.append(instruction_kwargs)
        self.instructions.append(tuple(instruction_list))
        self._internal_logger.debug(f"Placeholder: Added instruction to buffer: {tuple(instruction_list)}")

    # --- 统一的门方法集合 ---
    # 所有门方法都通过调用 `add_gate` 来实现，只负责指令收集。
    
    # 1. 基础单比特门
    def x(self, qubit_index: int, **kwargs: Any):
        # [健壮性改进] 基础参数类型检查
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'x' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        self.add_gate("x", qubit_index, **kwargs)

    def y(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'y' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        self.add_gate("y", qubit_index, **kwargs)

    def z(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'z' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        self.add_gate("z", qubit_index, **kwargs)

    def h(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'h' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        self.add_gate("h", qubit_index, **kwargs)

    def s(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 's' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        self.add_gate("s", qubit_index, **kwargs)
    
    def sx(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'sx' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        self.add_gate("sx", qubit_index, **kwargs)

    def sdg(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'sdg' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        self.add_gate("sdg", qubit_index, **kwargs)

    def t_gate(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 't_gate' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        self.add_gate("t_gate", qubit_index, **kwargs)

    def tdg(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'tdg' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        self.add_gate("tdg", qubit_index, **kwargs)

    # 2. 基础参数化单比特旋转门
    def rx(self, qubit_index: int, theta: float, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'rx' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        if not isinstance(theta, (float, int)): self._internal_logger.warning(f"Placeholder: 'rx' gate: 'theta' must be numeric, got {type(theta).__name__}. Storing as is.")
        self.add_gate("rx", qubit_index, theta, **kwargs)

    def ry(self, qubit_index: int, theta: float, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'ry' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        if not isinstance(theta, (float, int)): self._internal_logger.warning(f"Placeholder: 'ry' gate: 'theta' must be numeric, got {type(theta).__name__}. Storing as is.")
        self.add_gate("ry", qubit_index, theta, **kwargs)

    def rz(self, qubit_index: int, phi: float, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'rz' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        if not isinstance(phi, (float, int)): self._internal_logger.warning(f"Placeholder: 'rz' gate: 'phi' must be numeric, got {type(phi).__name__}. Storing as is.")
        self.add_gate("rz", qubit_index, phi, **kwargs)

    def p_gate(self, qubit_index: int, lambda_angle: float, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'p_gate' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        if not isinstance(lambda_angle, (float, int)): self._internal_logger.warning(f"Placeholder: 'p_gate' gate: 'lambda_angle' must be numeric, got {type(lambda_angle).__name__}. Storing as is.")
        self.add_gate("p_gate", qubit_index, lambda_angle, **kwargs)

    def u3_gate(self, qubit_index: int, theta: float, phi: float, lambda_angle: float, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'u3_gate' gate: 'qubit_index' must be int, got {type(qubit_index).__name__}. Storing as is.")
        if not all(isinstance(a, (float, int)) for a in [theta, phi, lambda_angle]): self._internal_logger.warning(f"Placeholder: 'u3_gate' gate: angles must be numeric. Storing as is.")
        self.add_gate("u3_gate", qubit_index, theta, phi, lambda_angle, **kwargs)

    # 3. 基础多比特门
    def cnot(self, control: int, target: int, **kwargs: Any):
        if not all(isinstance(q, int) for q in [control, target]): self._internal_logger.warning(f"Placeholder: 'cnot' gate: control/target must be int. Storing as is.")
        self.add_gate("cnot", control, target, **kwargs)

    def cz(self, control: int, target: int, **kwargs: Any):
        if not all(isinstance(q, int) for q in [control, target]): self._internal_logger.warning(f"Placeholder: 'cz' gate: control/target must be int. Storing as is.")
        self.add_gate("cz", control, target, **kwargs)

    def swap(self, qubit1: int, qubit2: int, **kwargs: Any):
        if not all(isinstance(q, int) for q in [qubit1, qubit2]): self._internal_logger.warning(f"Placeholder: 'swap' gate: qubits must be int. Storing as is.")
        self.add_gate("swap", qubit1, qubit2, **kwargs)

    def toffoli(self, control_1: int, control_2: int, target: int, **kwargs: Any):
        if not all(isinstance(q, int) for q in [control_1, control_2, target]): self._internal_logger.warning(f"Placeholder: 'toffoli' gate: control/target must be int. Storing as is.")
        self.add_gate("toffoli", control_1, control_2, target, **kwargs)

    def fredkin(self, control: int, target_1: int, target_2: int, **kwargs: Any):
        if not all(isinstance(q, int) for q in [control, target_1, target_2]): self._internal_logger.warning(f"Placeholder: 'fredkin' gate: control/target must be int. Storing as is.")
        self.add_gate("fredkin", control, target_1, target_2, **kwargs)

    # 4. 高级受控门
    def cp(self, control: int, target: int, angle: float, **kwargs: Any):
        if not all(isinstance(q, int) for q in [control, target]): self._internal_logger.warning(f"Placeholder: 'cp' gate: control/target must be int. Storing as is.")
        if not isinstance(angle, (float, int)): self._internal_logger.warning(f"Placeholder: 'cp' gate: 'angle' must be numeric. Storing as is.")
        self.add_gate("cp", control, target, angle, **kwargs)

    def crx(self, control: int, target: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) for q in [control, target]): self._internal_logger.warning(f"Placeholder: 'crx' gate: control/target must be int. Storing as is.")
        if not isinstance(theta, (float, int)): self._internal_logger.warning(f"Placeholder: 'crx' gate: 'theta' must be numeric. Storing as is.")
        self.add_gate("crx", control, target, theta, **kwargs)

    def cry(self, control: int, target: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) for q in [control, target]): self._internal_logger.warning(f"Placeholder: 'cry' gate: control/target must be int. Storing as is.")
        if not isinstance(theta, (float, int)): self._internal_logger.warning(f"Placeholder: 'cry' gate: 'theta' must be numeric. Storing as is.")
        self.add_gate("cry", control, target, theta, **kwargs)

    def crz(self, control: int, target: int, phi: float, **kwargs: Any):
        if not all(isinstance(q, int) for q in [control, target]): self._internal_logger.warning(f"Placeholder: 'crz' gate: control/target must be int. Storing as is.")
        if not isinstance(phi, (float, int)): self._internal_logger.warning(f"Placeholder: 'crz' gate: 'phi' must be numeric. Storing as is.")
        self.add_gate("crz", control, target, phi, **kwargs)

    def controlled_u(self, control: int, target: int, u_matrix: List[List[complex]], name: str = "CU", **kwargs: Any):
        if not all(isinstance(q, int) for q in [control, target]): self._internal_logger.warning(f"Placeholder: 'controlled_u' gate: control/target must be int. Storing as is.")
        # [健壮性改进] 对 u_matrix 进行更详细的检查
        if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
                isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
                isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2):
            self._internal_logger.warning(f"Placeholder: 'controlled_u' gate: 'u_matrix' must be a 2x2 nested list of complex numbers. Got {type(u_matrix).__name__}. Storing as is.")
        self.add_gate("controlled_u", control, target, u_matrix, name=name, **kwargs)

    # 5. 高级参数化多比特门 (VQA常用)
    def rxx(self, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) for q in [qubit1, qubit2]): self._internal_logger.warning(f"Placeholder: 'rxx' gate: qubits must be int. Storing as is.")
        if not isinstance(theta, (float, int)): self._internal_logger.warning(f"Placeholder: 'rxx' gate: 'theta' must be numeric. Storing as is.")
        self.add_gate("rxx", qubit1, qubit2, theta, **kwargs)
    
    def ryy(self, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) for q in [qubit1, qubit2]): self._internal_logger.warning(f"Placeholder: 'ryy' gate: qubits must be int. Storing as is.")
        if not isinstance(theta, (float, int)): self._internal_logger.warning(f"Placeholder: 'ryy' gate: 'theta' must be numeric. Storing as is.")
        self.add_gate("ryy", qubit1, qubit2, theta, **kwargs)
        
    def rzz(self, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) for q in [qubit1, qubit2]): self._internal_logger.warning(f"Placeholder: 'rzz' gate: qubits must be int. Storing as is.")
        if not isinstance(theta, (float, int)): self._internal_logger.warning(f"Placeholder: 'rzz' gate: 'theta' must be numeric. Storing as is.")
        self.add_gate("rzz", qubit1, qubit2, theta, **kwargs)

    # 6. 高级多控制门 (算法常用)
    def mcz(self, controls: List[int], target: int, **kwargs: Any):
        # [健壮性改进] 对 controls 列表进行检查
        if not isinstance(controls, list) or not all(isinstance(q, int) for q in controls): self._internal_logger.warning(f"Placeholder: 'mcz' gate: 'controls' must be list of int. Storing as is.")
        if not isinstance(target, int): self._internal_logger.warning(f"Placeholder: 'mcz' gate: 'target' must be int. Storing as is.")
        self.add_gate("mcz", controls, target, **kwargs)

    def mcx(self, controls: List[int], target: int, **kwargs: Any):
        if not isinstance(controls, list) or not all(isinstance(q, int) for q in controls): self._internal_logger.warning(f"Placeholder: 'mcx' gate: 'controls' must be list of int. Storing as is.")
        if not isinstance(target, int): self._internal_logger.warning(f"Placeholder: 'mcx' gate: 'target' must be int. Storing as is.")
        self.add_gate("mcx", controls, target, **kwargs)

    def mcp(self, controls: List[int], target: int, angle: float, **kwargs: Any):
        if not isinstance(controls, list) or not all(isinstance(q, int) for q in controls): self._internal_logger.warning(f"Placeholder: 'mcp' gate: 'controls' must be list of int. Storing as is.")
        if not isinstance(target, int): self._internal_logger.warning(f"Placeholder: 'mcp' gate: 'target' must be int. Storing as is.")
        if not isinstance(angle, (float, int)): self._internal_logger.warning(f"Placeholder: 'mcp' gate: 'angle' must be numeric. Storing as is.")
        self.add_gate("mcp", controls, target, angle, **kwargs)

    def mcu(self, controls: List[int], target: int, u_matrix: List[List[complex]], name: str = "MCU", **kwargs: Any):
        if not isinstance(controls, list) or not all(isinstance(q, int) for q in controls): self._internal_logger.warning(f"Placeholder: 'mcu' gate: 'controls' must be list of int. Storing as is.")
        if not isinstance(target, int): self._internal_logger.warning(f"Placeholder: 'mcu' gate: 'target' must be int. Storing as is.")
        # [健壮性改进] 对 u_matrix 进行更详细的检查
        if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
                isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
                isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2):
            self._internal_logger.warning(f"Placeholder: 'mcu' gate: 'u_matrix' must be a 2x2 nested list of complex numbers. Got {type(u_matrix).__name__}. Storing as is.")
        self.add_gate("mcu", controls, target, u_matrix, name=name, **kwargs)

    def iswap(self, qubit1: int, qubit2: int, **kwargs: Any):
        if not all(isinstance(q, int) for q in [qubit1, qubit2]): self._internal_logger.warning(f"Placeholder: 'iswap' gate: qubits must be int. Storing as is.")
        self.add_gate("iswap", qubit1, qubit2, **kwargs)

    def ecr(self, qubit1: int, qubit2: int, **kwargs: Any):
        if not all(isinstance(q, int) for q in [qubit1, qubit2]): self._internal_logger.warning(f"Placeholder: 'ecr' gate: qubits must be int. Storing as is.")
        self.add_gate("ecr", qubit1, qubit2, **kwargs)

    def ecrdg(self, qubit1: int, qubit2: int, **kwargs: Any):
        if not all(isinstance(q, int) for q in [qubit1, qubit2]): self._internal_logger.warning(f"Placeholder: 'ecrdg' gate: qubits must be int. Storing as is.")
        self.add_gate("ecrdg", qubit1, qubit2, **kwargs)

    # 7. 非酉/特殊操作
    def measure(self, qubit_index: int, classical_register_index: Optional[int] = None, collapse_state: bool = True, **kwargs: Any):
        if not isinstance(qubit_index, int): self._internal_logger.warning(f"Placeholder: 'measure' operation: 'qubit_index' must be int. Storing as is.")
        if classical_register_index is not None and not isinstance(classical_register_index, int): self._internal_logger.warning(f"Placeholder: 'measure' operation: 'classical_register_index' must be int or None. Storing as is.")
        if not isinstance(collapse_state, bool): self._internal_logger.warning(f"Placeholder: 'measure' operation: 'collapse_state' must be bool. Storing as is.")
        # 直接调用 _add_raw_gate，因为这是特殊的非酉操作，不应再被 add_gate 进一步处理宏
        self._add_raw_gate("simulate_measurement", qubit_index, classical_register_index=classical_register_index, collapse_state=collapse_state, **kwargs)

    def quantum_subroutine(self, subroutine_circuit: Any, target_qubits: List[int], name: str = "subroutine", **kwargs: Any):
        # [健壮性改进] 对 subroutine_circuit 进行更详细的检查，虽然它是占位符，但应尽量模拟真实 QuantumCircuit
        if not isinstance(subroutine_circuit, (type(self), 'QuantumCircuit')): # 允许 QuantumCircuit 实例或自身占位符实例
             self._internal_logger.warning(f"Placeholder: 'quantum_subroutine': 'subroutine_circuit' should be a QuantumCircuit or Placeholder, got {type(subroutine_circuit).__name__}. Storing as is.")
        if not isinstance(target_qubits, list) or not all(isinstance(q, int) for q in target_qubits): self._internal_logger.warning(f"Placeholder: 'quantum_subroutine': 'target_qubits' must be list of int. Storing as is.")
        if not isinstance(name, str): self._internal_logger.warning(f"Placeholder: 'quantum_subroutine': 'name' must be str. Storing as is.")
        self._add_raw_gate("quantum_subroutine", subroutine_circuit, target_qubits, name=name, **kwargs)

    def apply_channel(self, channel_type: str, target_qubits: Union[int, List[int], None], params: Dict[str, Any], **kwargs: Any):
        if not isinstance(channel_type, str): self._internal_logger.warning(f"Placeholder: 'apply_channel': 'channel_type' must be str. Storing as is.")
        if target_qubits is not None and not isinstance(target_qubits, (int, list)): self._internal_logger.warning(f"Placeholder: 'apply_channel': 'target_qubits' must be int, list of int, or None. Storing as is.")
        if isinstance(target_qubits, list) and not all(isinstance(q, int) for q in target_qubits): self._internal_logger.warning(f"Placeholder: 'apply_channel': 'target_qubits' list must be int. Storing as is.")
        if not isinstance(params, dict): self._internal_logger.warning(f"Placeholder: 'apply_channel': 'params' must be dict. Storing as is.")
        self._add_raw_gate("apply_quantum_channel", channel_type, target_qubits, params=params, **kwargs)

    def barrier(self, *qubits: int, **kwargs: Any):
        # [健壮性改进] 对 qubits 参数进行检查
        if not all(isinstance(q, int) for q in qubits): self._internal_logger.warning(f"Placeholder: 'barrier': 'qubits' must be int. Storing as is.")
        self._add_raw_gate("barrier", *qubits, **kwargs)
    
    # --- 魔术方法 (Dunder methods) ---
    
    def __len__(self) -> int:
        """返回收集到的指令数量。"""
        return len(self.instructions)

    def __iter__(self):
        """使得占位符对象可以被迭代。"""
        return iter(self.instructions)

    def __str__(self) -> str:
        """提供一个人类可读的字符串表示。"""
        parts = [f"PlaceholderCircuit ({self.num_qubits} qubits, {len(self.instructions)} instructions):"]
        if self.description: parts.append(f"  Description: {self.description}")
        for i, instr in enumerate(self.instructions): parts.append(f"  [{i:02d}]: {instr}")
        return "\n".join(parts)

# ========================================================================
# --- [新架构] 量子算子定义库 (Quantum Operator Definition Library) ---
# ========================================================================



@dataclass(frozen=True)
class OperatorDefinition:
    """
    [最终完整版] 一个不可变的、自包含的数据结构，用于封装单个量子操作的完整定义。
    它作为整个系统中所有量子门的“单一事实来源”，确保了定义的一致性和可验证性。

    Attributes:
        name (str):
            操作的唯一名称 (e.g., 'h', 'cnot', 'rx')。
        num_qubits (int):
            操作作用的量子比特数。对于可变数量控制比特的门 (如 'mcx')，
            此字段设为 0。
        description (str):
            该操作的人类可读描述。

        unitary_matrix (Optional[List[List[complex]]]):
            对于无参数的门，这是其固定的、后端无关的酉矩阵 (以纯Python列表表示)。
            例如，Hadamard 门的 2x2 矩阵。

        unitary_generator (Optional[Callable[..., List[List[complex]]]]):
            对于参数化的门 (如 RX(theta))，这是一个接收参数并动态生成其酉矩阵的函数。
            此函数应返回一个后端无关的纯Python列表矩阵。

        decomposition_rule (Optional[Callable[..., None]]):
            一个函数，定义了如何将此操作分解为其他更基础的操作。
            该函数接收一个 QuantumCircuit 实例和门的参数作为输入。

        sv_kernel_name (Optional[str]):
            如果存在，这是 _StateVectorEntity 中用于高效执行此操作的优化内核方法的名称。

        dm_kernel_name (Optional[str]):
            如果存在，这是 _DensityMatrixEntity 中用于高效执行此操作的优化内核方法的名称。

        parameter_info (Optional[Tuple[Dict[str, Any], ...]]):
            一个元组，描述了每个参数的元数据，用于验证。
            每个字典可以包含 'name', 'type', 'validator' 等键。
    """
    name: str
    num_qubits: int
    description: str
    
    # --- 核心数学定义 (至少需要以下三者之一) ---
    unitary_matrix: Optional[List[List[complex]]] = field(default=None, repr=False)
    unitary_generator: Optional[Callable[..., List[List[complex]]]] = field(default=None, repr=False)
    decomposition_rule: Optional[Callable[..., None]] = field(default=None, repr=False)
    
    # --- 性能优化与元数据 ---
    sv_kernel_name: Optional[str] = field(default=None)
    dm_kernel_name: Optional[str] = field(default=None)
    parameter_info: Optional[Tuple[Dict[str, Any], ...]] = field(default_factory=tuple)

    def __post_init__(self):
        """
        在对象创建后进行严格的自检，确保定义的有效性和一致性。
        """
        # 1. 基础类型验证
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("Operator 'name' must be a non-empty string.")
        if not isinstance(self.num_qubits, int) or self.num_qubits < 0:
            raise TypeError("Operator 'num_qubits' must be a non-negative integer.")
        if not isinstance(self.description, str):
            raise TypeError("Operator 'description' must be a string.")

        # 2. 确保至少存在一种数学定义
        if self.unitary_matrix is None and self.unitary_generator is None and self.decomposition_rule is None:
            raise ValueError(f"OperatorDefinition for '{self.name}' must provide at least one of: 'unitary_matrix', 'unitary_generator', or 'decomposition_rule'.")

        # 3. 验证 unitary_matrix 的格式 (如果是静态定义的)
        if self.unitary_matrix is not None:
            if not isinstance(self.unitary_matrix, list):
                raise TypeError(f"Operator '{self.name}': 'unitary_matrix' must be a list of lists.")
            if self.num_qubits > 0:
                expected_dim = 1 << self.num_qubits
                if (len(self.unitary_matrix) != expected_dim or 
                    not all(isinstance(row, list) and len(row) == expected_dim for row in self.unitary_matrix)):
                    raise ValueError(f"Operator '{self.name}': 'unitary_matrix' shape does not match num_qubits={self.num_qubits}.")

        # 4. 验证 callable 属性
        if self.unitary_generator is not None and not callable(self.unitary_generator):
            raise TypeError(f"Operator '{self.name}': 'unitary_generator' must be a callable function.")
        if self.decomposition_rule is not None and not callable(self.decomposition_rule):
            raise TypeError(f"Operator '{self.name}': 'decomposition_rule' must be a callable function.")

        # 5. 验证参数信息
        if self.parameter_info:
            if not isinstance(self.parameter_info, tuple) or not all(isinstance(p, dict) and 'name' in p and 'type' in p for p in self.parameter_info):
                raise TypeError(f"Operator '{self.name}': 'parameter_info' must be a tuple of dictionaries, each with 'name' and 'type' keys.")
            
            # 检查参数数量是否与 unitary_generator 匹配 (如果存在)
            if self.unitary_generator:
                import inspect
                try:
                    sig = inspect.signature(self.unitary_generator)
                    # 减去 'circuit' (如果存在)，因为生成器不接收它
                    num_gen_params = len(sig.parameters)
                    if len(self.parameter_info) != num_gen_params:
                        self._internal_logger.warning(
                            f"Operator '{self.name}': Number of parameters in 'parameter_info' ({len(self.parameter_info)}) does not match "
                            f"the signature of 'unitary_generator' ({num_gen_params}). This may cause issues."
                        )
                except (ValueError, TypeError): # inspect.signature 可能对某些 callable 失败
                    pass

    def get_unitary(self, *args: Any, **kwargs: Any) -> Optional[List[List[complex]]]:
        """
        [最终修复] 获取此操作的酉矩阵。
        此版本恢复到简单的逻辑，将参数直接传递给生成器。
        参数筛选的职责转移到调用者。
        """
        if self.unitary_matrix is not None:
            return self.unitary_matrix
        
        if self.unitary_generator is not None:
            try:
                # 恢复到直接调用，由调用者保证参数正确性
                return self.unitary_generator(*args, **kwargs)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Failed to generate unitary for '{self.name}' with args {args}, kwargs {kwargs}: {e}") from e

        if self.decomposition_rule is not None:
            # 此路径保持不变，用于真正的复合宏
            logging.getLogger(__name__).warning(f"Deriving unitary for '{self.name}' by expanding its decomposition. This is computationally expensive.")
            
            num_local_qubits = self.num_qubits
            if num_local_qubits == 0: 
                all_qubits_in_args = set()
                for arg in args:
                    if isinstance(arg, int): all_qubits_in_args.add(arg)
                    elif isinstance(arg, list) and all(isinstance(q, int) for q in arg): all_qubits_in_args.update(arg)
                num_local_qubits = len(all_qubits_in_args)

            if num_local_qubits == 0 and args:
                 num_local_qubits = 1 

            temp_circuit = QuantumCircuit(num_local_qubits)
            self.decomposition_rule(temp_circuit, *args, **kwargs)
            return get_effective_unitary(temp_circuit)
            
        return None
class QuantumOperatorLibrary:
    """
    一个静态类，作为所有量子操作定义的中央注册中心和权威来源。
    """
    _registry: Dict[str, OperatorDefinition] = {}
    _is_initialized: bool = False
    _internal_logger = logging.getLogger(f"{__name__}.QuantumOperatorLibrary")

    @classmethod
    def register(cls, definition: OperatorDefinition):
        """
        向库中注册一个新的算子定义。此方法应在库初始化期间调用。
        """
        if not cls._is_initialized:
             # 如果库尚未正式初始化，直接注册
             if not isinstance(definition, OperatorDefinition):
                 raise TypeError("Only OperatorDefinition instances can be registered.")
             
             if definition.name in cls._registry:
                 cls._internal_logger.warning(f"Operator '{definition.name}' is being re-registered during initialization. Overwriting previous definition.")
             
             cls._registry[definition.name] = definition
             cls._internal_logger.debug(f"Operator '{definition.name}' has been registered in the library.")
        else:
            # 如果库已初始化，则不允许动态注册，以保证库的稳定性
            cls._internal_logger.error("Cannot register new operators after the library has been initialized.")
            raise RuntimeError("The QuantumOperatorLibrary is locked after initialization.")

    @classmethod
    def get(cls, name: str) -> Optional[OperatorDefinition]:
        """
        从库中安全地检索一个算子定义。
        """
        if not cls._is_initialized:
            # 确保在任何组件尝试使用它之前，库都已填充完毕
            cls._initialize_library()
        return cls._registry.get(name)

    @classmethod
    def get_all_names(cls) -> List[str]:
        """
        返回所有已注册算子的名称列表。
        """
        if not cls._is_initialized:
            cls._initialize_library()
        return list(cls._registry.keys())

    @classmethod
    def _initialize_library(cls):
        """
        [核心初始化方法] 调用外部函数来填充操作符库。
        此方法确保库只被填充一次。
        """
        if cls._is_initialized:
            return
        
        cls._internal_logger.info("Initializing the Quantum Operator Library...")
        
        # 调用外部函数来执行所有注册操作
        _populate_operator_library()
        
        cls._is_initialized = True
        cls._internal_logger.info(f"Quantum Operator Library initialized successfully with {len(cls._registry)} operators.")


def _populate_operator_library():
    """
    [核心知识库 - 最终完整版] 定义所有内置的量子操作，并将它们注册到 QuantumOperatorLibrary。
    此函数在库首次被访问时由 _initialize_library 调用一次，作为所有量子门定义的唯一事实来源。
    """
    import math
    import cmath

    # ========================================================================
    # --- 1. 辅助函数：分解规则 (Decomposition Rules) ---
    #    用于定义复合门如何由其他门构成。
    # ========================================================================

    def _decompose_swap(circuit: 'QuantumCircuit', q1: int, q2: int, **kwargs):
        circuit.cnot(q1, q2, **kwargs)
        circuit.cnot(q2, q1, **kwargs)
        circuit.cnot(q1, q2, **kwargs)

    def _decompose_fredkin(circuit: 'QuantumCircuit', c: int, t1: int, t2: int, **kwargs):
        circuit.cnot(t2, t1, **kwargs)
        circuit.toffoli(c, t1, t2, **kwargs)
        circuit.cnot(t2, t1, **kwargs)

    def _decompose_iswap(circuit: 'QuantumCircuit', q1: int, q2: int, **kwargs):
        angle = -math.pi / 2.0
        circuit.rxx(q1, q2, angle, **kwargs)
        circuit.ryy(q1, q2, angle, **kwargs)

    def _decompose_ecr(circuit: 'QuantumCircuit', q1: int, q2: int, **kwargs):
        PI_HALF = math.pi / 2.0
        circuit.rz(q1, PI_HALF, **kwargs)
        circuit.rx(q1, PI_HALF, **kwargs)
        circuit.cnot(q1, q2, **kwargs)
        circuit.rx(q2, -PI_HALF, **kwargs)
        circuit.rz(q1, -PI_HALF, **kwargs)

    def _decompose_ecrdg(circuit: 'QuantumCircuit', q1: int, q2: int, **kwargs):
        PI_HALF = math.pi / 2.0
        circuit.rz(q1, PI_HALF, **kwargs)
        circuit.rx(q2, PI_HALF, **kwargs)
        circuit.cnot(q1, q2, **kwargs)
        circuit.rx(q1, -PI_HALF, **kwargs)
        circuit.rz(q1, -PI_HALF, **kwargs)
    
    # [FIX] 为可变比特数原子宏添加分解规则，指向其自身的宏定义
    # 这允许 get_unitary 通过在临时电路上展开来计算其矩阵
    _define_mcz_macro = globals().get('_define_mcz_macro')
    _define_mcx_macro = globals().get('_define_mcx_macro')
    _define_mcp_macro = globals().get('_define_mcp_macro')
    _define_mcu_macro = globals().get('_define_mcu_macro')


    # ========================================================================
    # --- 2. 辅助函数：酉矩阵生成器 (Unitary Generators) ---
    #    用于定义参数化门的矩阵。
    # ========================================================================

    def _gen_rx_matrix(theta: float):
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        return [[c, -1j * s], [-1j * s, c]]

    def _gen_ry_matrix(theta: float):
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        return [[c, -s], [s, c]]

    def _gen_rz_matrix(phi: float):
        phase_neg, phase_pos = cmath.exp(-1j * phi / 2.0), cmath.exp(1j * phi / 2.0)
        return [[phase_neg, 0], [0, phase_pos]]

    def _gen_p_gate_matrix(lambda_angle: float):
        return [[1, 0], [0, cmath.exp(1j * lambda_angle)]]

    def _gen_u3_matrix(theta: float, phi: float, lambda_angle: float):
        c_half, s_half = math.cos(theta/2), math.sin(theta/2)
        return [
            [c_half, -cmath.exp(1j * lambda_angle) * s_half],
            [cmath.exp(1j * phi) * s_half, cmath.exp(1j * (phi + lambda_angle)) * c_half]
        ]
    
    def _gen_cp_matrix(angle: float):
        return [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,cmath.exp(1j*angle)]]

    # [FIX] 新增受控旋转门的矩阵生成器
    def _gen_crx_matrix(theta: float):
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        s_factor = -1j * s
        return [[1,0,0,0], [0,1,0,0], [0,0,c,s_factor], [0,0,s_factor,c]]

    def _gen_cry_matrix(theta: float):
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        return [[1,0,0,0], [0,1,0,0], [0,0,c,-s], [0,0,s,c]]
        
    def _gen_crz_matrix(phi: float):
        phase_neg, phase_pos = cmath.exp(-1j * phi / 2.0), cmath.exp(1j * phi / 2.0)
        return [[1,0,0,0], [0,1,0,0], [0,0,phase_neg,0], [0,0,0,phase_pos]]
    
    def _gen_controlled_u_matrix(u_matrix: List[List[complex]], name: str = "CU"):
        u00, u01 = u_matrix[0]
        u10, u11 = u_matrix[1]
        return [[1,0,0,0], [0,1,0,0], [0,0,u00,u01], [0,0,u10,u11]]
    # [FIX END]

    def _gen_rxx_matrix(theta: float):
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        s_factor = -1j * s
        return [[c,0,0,s_factor], [0,c,s_factor,0], [0,s_factor,c,0], [s_factor,0,0,c]]

    def _gen_ryy_matrix(theta: float):
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        s_pos, s_neg = 1j * s, -1j * s
        return [[c,0,0,s_pos], [0,c,s_neg,0], [0,s_neg,c,0], [s_pos,0,0,c]]

    def _gen_rzz_matrix(theta: float):
        phase_even, phase_odd = cmath.exp(-1j*theta/2.0), cmath.exp(1j*theta/2.0)
        return [[phase_even,0,0,0], [0,phase_odd,0,0], [0,0,phase_odd,0], [0,0,0,phase_even]]

    # ========================================================================
    # --- 3. 注册所有量子门 (Register All Gates) ---
    # ========================================================================

    # --- 组 A: 基础单比特门 ---
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    QuantumOperatorLibrary.register(OperatorDefinition('h', 1, "Hadamard gate", unitary_matrix=[[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]], sv_kernel_name='_apply_h_kernel_sv', dm_kernel_name='_apply_h_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('x', 1, "Pauli-X gate", unitary_matrix=[[0,1],[1,0]], sv_kernel_name='_apply_x_kernel_sv', dm_kernel_name='_apply_x_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('y', 1, "Pauli-Y gate", unitary_matrix=[[0,-1j],[1j,0]], sv_kernel_name='_apply_y_kernel_sv', dm_kernel_name='_apply_y_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('z', 1, "Pauli-Z gate", unitary_matrix=[[1,0],[0,-1]], sv_kernel_name='_apply_z_kernel_sv', dm_kernel_name='_apply_z_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('s', 1, "Phase gate (S)", unitary_matrix=[[1,0],[0,1j]], sv_kernel_name='_apply_s_kernel_sv', dm_kernel_name='_apply_s_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('sdg', 1, "S-dagger gate", unitary_matrix=[[1,0],[0,-1j]], sv_kernel_name='_apply_sdg_kernel_sv', dm_kernel_name='_apply_sdg_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('t_gate', 1, "T gate", unitary_matrix=[[1,0],[0,cmath.exp(1j*math.pi/4)]], sv_kernel_name='_apply_t_gate_kernel_sv', dm_kernel_name='_apply_t_gate_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('tdg', 1, "T-dagger gate", unitary_matrix=[[1,0],[0,cmath.exp(-1j*math.pi/4)]], sv_kernel_name='_apply_tdg_kernel_sv', dm_kernel_name='_apply_tdg_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('sx', 1, "Sqrt(X) gate", unitary_matrix=[[(1+1j)/2, (1-1j)/2],[(1-1j)/2, (1+1j)/2]], sv_kernel_name='_apply_sx_kernel_sv'))

    # --- 组 B: 参数化单比特门 ---
    QuantumOperatorLibrary.register(OperatorDefinition('rx', 1, "Rotation around X-axis", unitary_generator=_gen_rx_matrix, sv_kernel_name='_apply_rx_kernel_sv', dm_kernel_name='_apply_rx_kernel_dm', parameter_info=({'name': 'theta', 'type': float},)))
    QuantumOperatorLibrary.register(OperatorDefinition('ry', 1, "Rotation around Y-axis", unitary_generator=_gen_ry_matrix, sv_kernel_name='_apply_ry_kernel_sv', dm_kernel_name='_apply_ry_kernel_dm', parameter_info=({'name': 'theta', 'type': float},)))
    QuantumOperatorLibrary.register(OperatorDefinition('rz', 1, "Rotation around Z-axis", unitary_generator=_gen_rz_matrix, sv_kernel_name='_apply_rz_kernel_sv', dm_kernel_name='_apply_rz_kernel_dm', parameter_info=({'name': 'phi', 'type': float},)))
    QuantumOperatorLibrary.register(OperatorDefinition('p_gate', 1, "Phase gate", unitary_generator=_gen_p_gate_matrix, sv_kernel_name='_apply_p_gate_kernel_sv', dm_kernel_name='_apply_p_gate_kernel_dm', parameter_info=({'name': 'lambda_angle', 'type': float},)))
    QuantumOperatorLibrary.register(OperatorDefinition('u3_gate', 1, "Generic U3 gate", unitary_generator=_gen_u3_matrix, sv_kernel_name='_apply_u3_gate_kernel_sv', parameter_info=({'name': 'theta', 'type': float}, {'name': 'phi', 'type': float}, {'name': 'lambda_angle', 'type': float})))

    # --- 组 C: 基础多比特门 ---
    QuantumOperatorLibrary.register(OperatorDefinition('cnot', 2, "Controlled-NOT gate", unitary_matrix=[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], sv_kernel_name='_apply_cnot_kernel_sv', dm_kernel_name='_apply_cnot_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('cz', 2, "Controlled-Z gate", unitary_matrix=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], sv_kernel_name='_apply_cz_kernel_sv', dm_kernel_name='_apply_cz_kernel_dm'))
    QuantumOperatorLibrary.register(OperatorDefinition('swap', 2, "SWAP gate", decomposition_rule=_decompose_swap))
    QuantumOperatorLibrary.register(OperatorDefinition('iswap', 2, "iSWAP gate", decomposition_rule=_decompose_iswap))
    QuantumOperatorLibrary.register(OperatorDefinition('ecr', 2, "ECR gate", decomposition_rule=_decompose_ecr))
    QuantumOperatorLibrary.register(OperatorDefinition('ecrdg', 2, "ECR-dagger gate", decomposition_rule=_decompose_ecrdg))

    # --- 组 D: 参数化多比特门 ---
    QuantumOperatorLibrary.register(OperatorDefinition('cp', 2, "Controlled-Phase gate", unitary_generator=_gen_cp_matrix, sv_kernel_name='_apply_cp_kernel_sv', dm_kernel_name='_apply_cp_kernel_dm', parameter_info=({'name': 'angle', 'type': float},)))
    # [FIX] 添加 unitary_generator
    QuantumOperatorLibrary.register(OperatorDefinition('crx', 2, "Controlled-RX gate", unitary_generator=_gen_crx_matrix, sv_kernel_name='_apply_crx_kernel_sv', dm_kernel_name='_apply_crx_kernel_dm', parameter_info=({'name': 'theta', 'type': float},)))
    QuantumOperatorLibrary.register(OperatorDefinition('cry', 2, "Controlled-RY gate", unitary_generator=_gen_cry_matrix, sv_kernel_name='_apply_cry_kernel_sv', dm_kernel_name='_apply_cry_kernel_dm', parameter_info=({'name': 'theta', 'type': float},)))
    QuantumOperatorLibrary.register(OperatorDefinition('crz', 2, "Controlled-RZ gate", unitary_generator=_gen_crz_matrix, sv_kernel_name='_apply_crz_kernel_sv', dm_kernel_name='_apply_crz_kernel_dm', parameter_info=({'name': 'phi', 'type': float},)))
    QuantumOperatorLibrary.register(OperatorDefinition('controlled_u', 2, "Generic Controlled-U gate", unitary_generator=_gen_controlled_u_matrix, sv_kernel_name='_apply_controlled_u_kernel_sv', parameter_info=({'name': 'u_matrix', 'type': list}, {'name': 'name', 'type': str})))
    # [FIX END]
    QuantumOperatorLibrary.register(OperatorDefinition('rxx', 2, "RXX entanglement gate", unitary_generator=_gen_rxx_matrix, sv_kernel_name='_apply_rxx_kernel_sv', parameter_info=({'name': 'theta', 'type': float},)))
    QuantumOperatorLibrary.register(OperatorDefinition('ryy', 2, "RYY entanglement gate", unitary_generator=_gen_ryy_matrix, sv_kernel_name='_apply_ryy_kernel_sv', parameter_info=({'name': 'theta', 'type': float},)))
    QuantumOperatorLibrary.register(OperatorDefinition('rzz', 2, "RZZ entanglement gate", unitary_generator=_gen_rzz_matrix, sv_kernel_name='_apply_rzz_kernel_sv', parameter_info=({'name': 'theta', 'type': float},)))
    
    # --- 组 E: 高级多控制门 (原子门，有优化内核) ---
    # [FIX] 为 toffoli 添加静态酉矩阵
    toffoli_matrix = [
        [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,1,0]
    ]
    QuantumOperatorLibrary.register(OperatorDefinition('toffoli', 3, "Toffoli (CCX) gate", unitary_matrix=toffoli_matrix, sv_kernel_name='_apply_toffoli_kernel_sv'))
    
    
    # [FIX] 为可变比特数宏添加 decomposition_rule 以通过验证
    QuantumOperatorLibrary.register(OperatorDefinition('fredkin', 3, "Fredkin (CSWAP) gate", decomposition_rule=_decompose_fredkin))
    QuantumOperatorLibrary.register(OperatorDefinition('mcx', 0, "Multi-Controlled-X gate", decomposition_rule=_define_mcx_macro, sv_kernel_name='_apply_mcx_kernel_sv', parameter_info=({'name': 'controls', 'type': list}, {'name': 'target', 'type': int})))
    QuantumOperatorLibrary.register(OperatorDefinition('mcz', 0, "Multi-Controlled-Z gate", decomposition_rule=_define_mcz_macro, sv_kernel_name='_apply_mcz_kernel_sv', parameter_info=({'name': 'controls', 'type': list}, {'name': 'target', 'type': int})))
    QuantumOperatorLibrary.register(OperatorDefinition('mcp', 0, "Multi-Controlled-Phase gate", decomposition_rule=_define_mcp_macro, sv_kernel_name='_apply_mcp_kernel_sv', parameter_info=({'name': 'controls', 'type': list}, {'name': 'target', 'type': int}, {'name': 'angle', 'type': float})))
    QuantumOperatorLibrary.register(OperatorDefinition('mcu', 0, "Multi-Controlled-U gate", decomposition_rule=_define_mcu_macro, sv_kernel_name='_apply_mcu_kernel_sv', parameter_info=({'name': 'controls', 'type': list}, {'name': 'target', 'type': int}, {'name': 'u_matrix', 'type': list})))
    

    # --- 组 F: 特殊/非酉操作 (没有酉矩阵或分解，但需要被识别) ---
    QuantumOperatorLibrary.register(OperatorDefinition('barrier', 0, "Barrier", decomposition_rule=lambda circuit, *qubits, **kwargs: None)) # 空分解
    # [FIX] 为 apply_unitary 添加一个空的分解规则以通过验证
    QuantumOperatorLibrary.register(OperatorDefinition('apply_unitary', 0, "Apply arbitrary unitary", decomposition_rule=lambda *args, **kwargs: None, sv_kernel_name='_apply_apply_unitary_kernel_sv'))
    



@dataclass
class _InstructionBuffer:
    """
    [内部基类] 一个简单的数据结构，其唯一职责是存储和管理一个指令列表。
    它提供了所有门方法的存根(stub)，这些方法都委托给一个必须由子类实现的
    `add_gate` 方法。这确保了所有门操作都有一个统一的入口点。
    """
    num_qubits: int
    instructions: List[Tuple[Any, ...]] = field(default_factory=list)
    description: Optional[str] = None
    _internal_logger: logging.Logger = field(default_factory=lambda: logging.getLogger(f"{__name__}._InstructionBuffer"), repr=False, init=False)

    def __post_init__(self):
        if not isinstance(self.num_qubits, int) or self.num_qubits < 0:
            raise ValueError("'num_qubits' must be a non-negative integer.")
    
    def _add_raw_gate(self, gate_name: str, *args: Any, **kwargs: Any):
        """直接向指令列表中添加一个元组，不进行任何处理。"""
        instruction_list: List[Any] = [gate_name] + list(args)
        if kwargs:
            instruction_list.append(kwargs)
        self.instructions.append(tuple(instruction_list))

    def add_gate(self, gate_name: str, *args: Any, **kwargs: Any):
        """[抽象占位符] 子类必须覆盖此方法以实现门添加逻辑。"""
        raise NotImplementedError("Subclasses of _InstructionBuffer must implement the 'add_gate' method.")

    # --- 统一的门方法集合 (委托给 self.add_gate) ---
    def x(self, qubit_index: int, **kwargs: Any): self.add_gate("x", qubit_index, **kwargs)
    def y(self, qubit_index: int, **kwargs: Any): self.add_gate("y", qubit_index, **kwargs)
    def z(self, qubit_index: int, **kwargs: Any): self.add_gate("z", qubit_index, **kwargs)
    def h(self, qubit_index: int, **kwargs: Any): self.add_gate("h", qubit_index, **kwargs)
    def s(self, qubit_index: int, **kwargs: Any): self.add_gate("s", qubit_index, **kwargs)
    def sx(self, qubit_index: int, **kwargs: Any): self.add_gate("sx", qubit_index, **kwargs)
    def sdg(self, qubit_index: int, **kwargs: Any): self.add_gate("sdg", qubit_index, **kwargs)
    def t_gate(self, qubit_index: int, **kwargs: Any): self.add_gate("t_gate", qubit_index, **kwargs)
    def tdg(self, qubit_index: int, **kwargs: Any): self.add_gate("tdg", qubit_index, **kwargs)
    def rx(self, qubit_index: int, theta: float, **kwargs: Any): self.add_gate("rx", qubit_index, theta, **kwargs)
    def ry(self, qubit_index: int, theta: float, **kwargs: Any): self.add_gate("ry", qubit_index, theta, **kwargs)
    def rz(self, qubit_index: int, phi: float, **kwargs: Any): self.add_gate("rz", qubit_index, phi, **kwargs)
    def p_gate(self, qubit_index: int, lambda_angle: float, **kwargs: Any): self.add_gate("p_gate", qubit_index, lambda_angle, **kwargs)
    def u3_gate(self, qubit_index: int, theta: float, phi: float, lambda_angle: float, **kwargs: Any): self.add_gate("u3_gate", qubit_index, theta, phi, lambda_angle, **kwargs)
    def cnot(self, control: int, target: int, **kwargs: Any): self.add_gate("cnot", control, target, **kwargs)
    def cz(self, control: int, target: int, **kwargs: Any): self.add_gate("cz", control, target, **kwargs)
    def swap(self, qubit1: int, qubit2: int, **kwargs: Any): self.add_gate("swap", qubit1, qubit2, **kwargs)
    def toffoli(self, c1: int, c2: int, target: int, **kwargs: Any): self.add_gate("toffoli", c1, c2, target, **kwargs)
    def fredkin(self, c: int, t1: int, t2: int, **kwargs: Any): self.add_gate("fredkin", c, t1, t2, **kwargs)
    def cp(self, c: int, t: int, angle: float, **kwargs: Any): self.add_gate("cp", c, t, angle, **kwargs)
    def crx(self, c: int, t: int, theta: float, **kwargs: Any): self.add_gate("crx", c, t, theta, **kwargs)
    def cry(self, c: int, t: int, theta: float, **kwargs: Any): self.add_gate("cry", c, t, theta, **kwargs)
    def crz(self, c: int, t: int, phi: float, **kwargs: Any): self.add_gate("crz", c, t, phi, **kwargs)
    def controlled_u(self, c: int, t: int, u: List[List[complex]], **kwargs: Any): self.add_gate("controlled_u", c, t, u, **kwargs)
    def rxx(self, q1: int, q2: int, theta: float, **kwargs: Any): self.add_gate("rxx", q1, q2, theta, **kwargs)
    def ryy(self, q1: int, q2: int, theta: float, **kwargs: Any): self.add_gate("ryy", q1, q2, theta, **kwargs)
    def rzz(self, q1: int, q2: int, theta: float, **kwargs: Any): self.add_gate("rzz", q1, q2, theta, **kwargs)
    def mcx(self, cs: List[int], t: int, **kwargs: Any): self.add_gate("mcx", cs, t, **kwargs)
    def mcz(self, cs: List[int], t: int, **kwargs: Any): self.add_gate("mcz", cs, t, **kwargs)
    def mcp(self, cs: List[int], t: int, angle: float, **kwargs: Any): self.add_gate("mcp", cs, t, angle, **kwargs)
    def mcu(self, cs: List[int], t: int, u: List[List[complex]], **kwargs: Any): self.add_gate("mcu", cs, t, u, **kwargs)
    def iswap(self, q1: int, q2: int, **kwargs: Any): self.add_gate("iswap", q1, q2, **kwargs)
    def ecr(self, q1: int, q2: int, **kwargs: Any): self.add_gate("ecr", q1, q2, **kwargs)
    def ecrdg(self, q1: int, q2: int, **kwargs: Any): self.add_gate("ecrdg", q1, q2, **kwargs)

    # --- 特殊操作 (直接使用 _add_raw_gate) ---
    def measure(self, qubit_index: int, classical_register_index: Optional[int] = None, **kwargs: Any): self._add_raw_gate("simulate_measurement", qubit_index, classical_register_index=classical_register_index, **kwargs)
    def quantum_subroutine(self, sub_circuit: Any, targets: List[int], **kwargs: Any): self._add_raw_gate("quantum_subroutine", sub_circuit, targets, **kwargs)
    def apply_channel(self, type: str, targets: Any, params: Dict, **kwargs: Any): self._add_raw_gate("apply_quantum_channel", type, targets, params=params, **kwargs)
    def barrier(self, *qubits: int, **kwargs: Any): self._add_raw_gate("barrier", *qubits, **kwargs)

    # --- 魔术方法 ---
    def __len__(self) -> int: return len(self.instructions)
    def __iter__(self): return iter(self.instructions)
    def __str__(self) -> str:
        parts = [f"InstructionBuffer ({self.num_qubits} qubits, {len(self.instructions)} instructions):"]
        if self.description: parts.append(f"  Description: {self.description}")
        for i, instr in enumerate(self.instructions): parts.append(f"  [{i:02d}]: {instr}")
        return "\n".join(parts)

@dataclass
class QuantumCircuit(_InstructionBuffer):
    """
    [最终完整版 - 基于 OperatorLibrary] 表示一个量子电路，包含一系列按顺序应用的量子门操作指令。

    此类继承自 _InstructionBuffer，并实现了核心的 `add_gate` 逻辑，该逻辑完全由
    QuantumOperatorLibrary 驱动。这确保了所有门操作的定义、分解和记录都遵循一个
    统一、集中和可验证的规则。

    核心特性:
    - **单一入口点**: 所有门方法 (x, h, cnot, etc.) 都通过 `add_gate` 路由。
    - **定义驱动**: `add_gate` 的行为（分解宏或记录原子操作）由 QuantumOperatorLibrary 决定。
    - **健壮性**: 对所有门方法的参数进行类型和范围验证。
    - **清晰性**: 代码结构清晰，职责分离。
    """
    # 继承字段: num_qubits, instructions, description
    
    # 覆盖从父类继承的日志器，使用正确的类名
    _internal_logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(f"{__name__}.QuantumCircuit"), 
        repr=False, 
        init=False
    )
    
    def add_gate(self, gate_name: str, *args: Any, condition: Optional[Tuple[int, int]] = None, **kwargs: Any):
        """
        [最终统一版 - 职责单一化] 向电路中添加一个门操作指令。
        此方法的唯一职责是验证门的存在性，并将指令以标准格式记录到内部指令列表中。
        宏展开的职责被完全转移到执行引擎（如 _StateVectorEntity）。
        """
        log_prefix = f"QuantumCircuit.add_gate(Op: '{gate_name}')"
        
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(gate_name, str) or not gate_name.strip():
            self._internal_logger.error(f"[{log_prefix}] Invalid gate_name parameter ('{gate_name}'). It must be a non-empty string.")
            raise ValueError("Gate name must be a non-empty string.")
            
        instruction_kwargs = dict(kwargs) 
        if condition is not None:
            if not (isinstance(condition, tuple) and len(condition) == 2 and
                    isinstance(condition[0], int) and isinstance(condition[1], int)):
                self._internal_logger.error(f"[{log_prefix}] Invalid condition format ('{condition}'). Must be a tuple of two integers.")
                raise TypeError("Condition must be a tuple of two integers: (classical_register_index, expected_value).")
            instruction_kwargs['condition'] = condition

        # --- 步骤 2: 从库中查询操作定义以进行验证 ---
        op_def = QuantumOperatorLibrary.get(gate_name)

        if op_def is None:
            self._internal_logger.warning(
                f"[{log_prefix}] Gate '{gate_name}' is not defined in the QuantumOperatorLibrary. "
                "Adding as a raw, opaque instruction. Ensure your execution engine knows how to handle it."
            )
        else:
            # --- 步骤 3: 参数验证 (基于库中的定义) ---
            all_qubits_in_args = []
            for arg in args:
                if isinstance(arg, int):
                    all_qubits_in_args.append(arg)
                elif isinstance(arg, list) and all(isinstance(q, int) for q in arg):
                    all_qubits_in_args.extend(arg)
            
            for q_idx in all_qubits_in_args:
                if not (0 <= q_idx < self.num_qubits):
                    self._internal_logger.error(f"[{log_prefix}] Qubit index {q_idx} for gate '{gate_name}' is out of range for this {self.num_qubits}-qubit circuit.")
                    raise ValueError(f"Qubit index {q_idx} for gate '{gate_name}' is out of range for this {self.num_qubits}-qubit circuit.")
        
        # --- 步骤 4: [核心修改] 无论门是原子还是复合，都直接记录原始指令 ---
        self._internal_logger.debug(f"[{log_prefix}] Adding raw instruction for '{gate_name}' to the circuit.")
        self._add_raw_gate(gate_name, *args, **instruction_kwargs)
    def __str__(self) -> str:
        """
        提供一个人类可读的、更美观的电路字符串表示。
        """
        parts = [f"QuantumCircuit ({self.num_qubits} qubits, {len(self.instructions)} instructions):"]
        if self.description:
            parts.append(f"  Description: {self.description}")
        
        for i, instr in enumerate(self.instructions):
            # 辅助函数：解析指令元组
            def _parse_instr(instruction_tuple):
                name = instruction_tuple[0]
                has_kwargs = len(instruction_tuple) > 1 and isinstance(instruction_tuple[-1], dict)
                args = instruction_tuple[1:-1] if has_kwargs else instruction_tuple[1:]
                kwargs = instruction_tuple[-1] if has_kwargs else {}
                return name, args, kwargs

            op_name, op_args, op_kwargs = _parse_instr(instr)
            
            condition_str = ""
            if 'condition' in op_kwargs:
                cr_idx, exp_val = op_kwargs['condition']
                condition_str = f" IF C[{cr_idx}]=={exp_val}"
                kwargs_for_display = {k: v for k, v in op_kwargs.items() if k != 'condition'}
            else:
                kwargs_for_display = op_kwargs

            # 辅助函数：格式化参数以进行打印
            def format_arg(arg: Any) -> str:
                if isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], (list, tuple)):
                    try:
                        rows, cols = len(arg), len(arg[0])
                        return f"matrix(shape=({rows},{cols}))"
                    except (TypeError, IndexError):
                         return str(arg)
                if isinstance(arg, float):
                    return f"{arg:.4f}"
                if isinstance(arg, (_InstructionBuffer, QuantumCircuit)):
                    return f"SubCircuit({arg.description or 'unnamed'})"
                return str(arg)

            args_str = ", ".join(map(format_arg, op_args))
            kwargs_str = ", ".join(f"{k}={format_arg(v)}" for k, v in kwargs_for_display.items())
            
            params_str = ", ".join(filter(None, [args_str, kwargs_str]))
            
            parts.append(f"  [{i:02d}]: {op_name}({params_str}){condition_str}")
            
        return "\n".join(parts)
@dataclass(frozen=True, eq=False) # frozen=True 保证不可变性, eq=False 因为我们将自定义 __eq__ 和 __hash__
class TranspiledUnitary:
    """
    [最终修正增强版] 表示一个可逆的、经过编译的酉矩阵。

    它封装了用于高效模拟的酉矩阵，但同时保留了其来源、分解和
    优化过程的元数据，从而实现了“黑盒效率”与“白盒透明度”的结合。

    此类的实例是不可变的，并且可以安全地作为字典键或集合元素使用
    （基于其元数据进行哈希和比较，而不是基于其内部的酉矩阵）。
    """
    unitary_matrix: Any = field(compare=False, hash=False, repr=False) # 实际的酉矩阵，不参与比较、哈希或标准表示
    num_qubits: int # 该酉矩阵作用的局部量子比特数

    # --- 可逆性与追溯性元数据 ---
    source_circuit_hash: Optional[str] = field(default=None) # 原始电路的哈希，可追溯到源
    source_circuit_description: Optional[str] = field(default=None) # 原始电路的描述
    
    # 存储原始的/优化的门序列（作为 QuantumCircuit 对象）
    # 在需要展开时使用。不参与默认比较和哈希。
    optimized_gate_sequence: Optional['QuantumCircuit'] = field(default=None, compare=False, hash=False, repr=False)
    
    # 编译/优化过程的元数据
    optimizer_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        在对象初始化后进行全面的类型和值验证。
        此方法由 `dataclasses` 模块自动调用，确保每个实例都是有效的。
        """
        # --- 1. 验证核心属性 ---
        if not isinstance(self.num_qubits, int) or self.num_qubits < 0:
            raise ValueError(f"TranspiledUnitary: 'num_qubits' must be a non-negative integer, but got {self.num_qubits}.")
        
        # 验证 unitary_matrix 的维度是否与 num_qubits 匹配
        expected_dim = 1 << self.num_qubits
        matrix_dim = 0
        
        # 兼容 CuPy/NumPy 数组和 Python 列表
        if hasattr(self.unitary_matrix, 'shape'):
            shape = self.unitary_matrix.shape
            if len(shape) == 2 and shape[0] == shape[1]:
                matrix_dim = shape[0]
        elif isinstance(self.unitary_matrix, list) and self.unitary_matrix and isinstance(self.unitary_matrix[0], list):
            if len(self.unitary_matrix) == len(self.unitary_matrix[0]):
                matrix_dim = len(self.unitary_matrix)
        
        if matrix_dim != expected_dim:
            raise ValueError(
                f"TranspiledUnitary: Unitary matrix dimension ({matrix_dim}) does not match num_qubits ({self.num_qubits}). "
                f"Expected a {expected_dim}x{expected_dim} matrix."
            )

        # --- 2. 验证元数据属性 ---
        if self.source_circuit_hash is not None and not isinstance(self.source_circuit_hash, str):
            raise TypeError(f"TranspiledUnitary: 'source_circuit_hash' must be a string or None, but got {type(self.source_circuit_hash).__name__}.")
            
        if self.source_circuit_description is not None and not isinstance(self.source_circuit_description, str):
            raise TypeError(f"TranspiledUnitary: 'source_circuit_description' must be a string or None, but got {type(self.source_circuit_description).__name__}.")
            
        if self.optimized_gate_sequence is not None and not isinstance(self.optimized_gate_sequence, QuantumCircuit):
            raise TypeError(f"TranspiledUnitary: 'optimized_gate_sequence' must be a QuantumCircuit instance or None, but got {type(self.optimized_gate_sequence).__name__}.")
        
        if self.optimized_gate_sequence is not None and self.optimized_gate_sequence.num_qubits != self.num_qubits:
             raise ValueError(
                 f"TranspiledUnitary: The qubit count of 'optimized_gate_sequence' ({self.optimized_gate_sequence.num_qubits}) "
                 f"must match the TranspiledUnitary's 'num_qubits' ({self.num_qubits})."
             )
        
        if not isinstance(self.optimizer_metadata, dict):
            raise TypeError(f"TranspiledUnitary: 'optimizer_metadata' must be a dictionary, but got {type(self.optimizer_metadata).__name__}.")


    def get_equivalent_circuit(self) -> 'QuantumCircuit':
        """
        返回此 TranspiledUnitary 的等效 QuantumCircuit，以实现“白盒”可追溯性。

        此方法提供了从“黑盒”酉矩阵回到“白盒”门序列的关键能力。
        它的行为是分层的：
        1.  (最高优先级) 如果 `optimized_gate_sequence` 存在，它会返回这个
            经过优化的、人类可读的门序列。这提供了最完整的可追溯性。
        2.  (后备方案) 如果没有存储门序列，它会构建并返回一个只包含
            `('apply_unitary', ...)` 指令的简单电路。这虽然不如前者透明，
            但仍然保持了操作的量子线路表示。

        Returns:
            QuantumCircuit: 一个新的、独立的 QuantumCircuit 实例。
        """
        if self.optimized_gate_sequence is not None:
            # 深拷贝以确保返回的电路是独立的，可以被修改，而不会影响原始的 TranspiledUnitary 对象
            return copy.deepcopy(self.optimized_gate_sequence)
        
        # 如果没有存储优化门序列，则返回一个包含 'apply_unitary' 的简单电路
        circuit = QuantumCircuit(
            self.num_qubits,
            description=(
                f"Circuit from TranspiledUnitary({self.num_qubits} qubits, "
                f"source_hash='{self.source_circuit_hash[:8] if self.source_circuit_hash else 'N/A'}')"
            )
        )
        # add_gate 会将这个操作添加到电路的指令列表中
        circuit.add_gate('apply_unitary', self.unitary_matrix, list(range(self.num_qubits)))
        return circuit

    def __eq__(self, other: Any) -> bool:
        """
        自定义相等性比较。两个 TranspiledUnitary 对象相等当且仅当它们
        的可追溯元数据完全相同。内部的酉矩阵和优化门序列不参与比较。
        """
        if not isinstance(other, TranspiledUnitary):
            return NotImplemented
        
        # 比较所有被标记为 compare=True (默认) 的字段
        return (
            self.num_qubits == other.num_qubits and
            self.source_circuit_hash == other.source_circuit_hash and
            self.source_circuit_description == other.source_circuit_description and
            self.optimizer_metadata == other.optimizer_metadata
        )

    def __hash__(self) -> int:
        """
        自定义哈希计算。哈希值仅基于可追溯的元数据，确保了对象的可哈希性。
        """
        # 将可变的 optimizer_metadata 字典转换为不可变的 frozenset of items，使其可哈希
        try:
            metadata_hashable = frozenset(self.optimizer_metadata.items())
        except TypeError:
            # 如果字典的值是不可哈希的（例如列表），则使用字符串表示作为后备
            metadata_hashable = str(sorted(self.optimizer_metadata.items()))

        return hash((
            self.num_qubits,
            self.source_circuit_hash,
            self.source_circuit_description,
            metadata_hashable
        ))

    def __repr__(self) -> str:
        """
        提供一个清晰、简洁的字符串表示形式，用于调试和日志记录。
        有意地隐藏了庞大的酉矩阵和门序列。
        """
        parts = [
            f"num_qubits={self.num_qubits}",
        ]
        if self.source_circuit_description:
            parts.append(f"source_desc='{self.source_circuit_description}'")
        if self.source_circuit_hash:
            parts.append(f"source_hash='{self.source_circuit_hash[:12]}...'")
        if self.optimized_gate_sequence is not None:
            parts.append(f"has_gate_sequence=True (len={len(self.optimized_gate_sequence.instructions)})")
        if self.optimizer_metadata:
            parts.append(f"metadata_keys={list(self.optimizer_metadata.keys())}")
            
        return f"TranspiledUnitary({', '.join(parts)})"
# ========================================================================
# --- 4. [Project Bedrock] 纯Python后端，用于终极调试与稳定性验证 ---
# ========================================================================

class PurePythonBackend:
    """
    [健壮性改进版] 一个完全用纯Python列表和标准库实现的线性代数后端。
    用于在不依赖任何第三方库的情况下，验证核心量子算法的逻辑正确性。
    性能较低，但其健壮性和稳定性至关重要。
    
    数据结构:
    - 向量: Python的列表 (e.g., [1+0j, 0+0j])
    - 矩阵: Python的嵌套列表 (e.g., [[1+0j, 0+0j], [0+0j, 1+0j]])
    """
    
    import random # Python标准库的随机数生成
    import math   # Python标准库的数学函数
    import cmath  # Python标准库的复数数学函数
    import builtins # Python内置函数
    from typing import Optional, List, Tuple, Dict, Any, Union

    # 用于 Box-Muller 变换的缓存 (这是一个类属性，因为它是跨实例共享的)
    _rand_normal_cache: Optional[float] = None
    
    def __init__(self):
        """
        初始化 PurePythonBackend 实例。
        """
        # 内部日志器 (在实例初始化时定义，解决 NameError)
        self._internal_logger: logging.Logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._internal_logger.debug("PurePythonBackend initialized.")

    # --- 基础矩阵/向量创建 ---

    def create_matrix(self, rows: int, cols: int, value: complex = 0.0 + 0.0j) -> List[List[complex]]:
        """
        创建一个由嵌套列表表示的、指定大小和初始值的复数矩阵。

        Args:
            rows (int): 矩阵的行数。必须是非负整数。
            cols (int): 矩阵的列数。必须是非负整数。
            value (complex, optional):
                矩阵中每个元素的初始值。默认为 0.0 + 0.0j。

        Returns:
            List[List[complex]]:
                一个代表新创建矩阵的嵌套列表。

        Raises:
            ValueError: 如果 `rows` 或 `cols` 为负数。
            TypeError: 如果 `rows` 或 `cols` 不是整数，或 `value` 不是数值类型。
        """
        # --- 输入验证 ---
        if not isinstance(rows, int) or not isinstance(cols, int):
            self._internal_logger.error(f"create_matrix: Matrix dimensions (rows={rows}, cols={cols}) must be integers.")
            raise TypeError("Matrix dimensions (rows, cols) must be integers.")
        if rows < 0 or cols < 0:
            self._internal_logger.error(f"create_matrix: Matrix dimensions (rows={rows}, cols={cols}) cannot be negative.")
            raise ValueError("Matrix dimensions (rows, cols) cannot be negative.")
        if not isinstance(value, (complex, float, int)):
            self._internal_logger.error(f"create_matrix: Initial 'value' must be a numeric type, but got {type(value).__name__}.")
            raise TypeError("Initial 'value' must be a numeric type.")

        # --- 核心实现 ---
        # 使用列表推导式高效地创建嵌套列表。
        # 确保每个元素都显式转换为 complex 类型。
        return [[complex(value) for _ in range(cols)] for _ in range(rows)]

    def eye(self, dim: int, dtype=None) -> List[List[complex]]:
        """
        创建一个指定维度的单位矩阵 (Identity Matrix)。

        单位矩阵是一个方阵，其主对角线上的元素为 1.0 + 0.0j，
        所有其他元素为 0.0 + 0.0j。
        此函数用于替代 `numpy.eye`。

        Args:
            dim (int):
                方阵的维度（行数或列数）。必须是非负整数。
            dtype (Any, optional):
                此参数为了与 NumPy API 保持兼容而被接受，但会被忽略。
                返回的矩阵元素类型始终是 Python 的 `complex`。

        Returns:
            List[List[complex]]:
                一个代表单位矩阵的嵌套列表。

        Raises:
            ValueError: 如果 `dim` 为负数。
            TypeError: 如果 `dim` 不是整数。
        """
        # --- 输入验证 ---
        if not isinstance(dim, int):
            self._internal_logger.error(f"eye: Dimension (dim={dim}) must be an integer.")
            raise TypeError("Dimension (dim) must be an integer.")
        if dim < 0:
            self._internal_logger.error(f"eye: Dimension (dim={dim}) cannot be negative.")
            raise ValueError("Dimension (dim) cannot be negative.")

        # --- 核心实现 ---
        # 1. 首先使用 create_matrix 创建一个全零的 dim x dim 矩阵。
        mat = self.create_matrix(dim, dim, value=0.0 + 0.0j)

        # 2. 然后，通过一个简单的循环，将主对角线上的元素设置为 1。
        for i in range(dim):
            mat[i][i] = 1.0 + 0.0j
            
        return mat

    def zeros(self, shape: Tuple[int, ...], dtype=None) -> Union[List[complex], List[List[complex]]]:
        """
        创建一个指定形状的全零向量或矩阵。

        此函数用于替代 `numpy.zeros`。根据输入 `shape` 元组的长度，
        它可以生成一维列表（向量）或嵌套列表（二维矩阵）。

        Args:
            shape (Tuple[int, ...]):
                一个元组，指定了输出数组的维度。
                - `(n,)`: 创建一个长度为 n 的一维向量。
                - `(m, n)`: 创建一个 m 行 n 列的二维矩阵。
                所有维度值必须是非负整数。
            dtype (Any, optional):
                此参数为了与 NumPy API 保持兼容而被接受，但会被忽略。
                返回的数组元素类型始终是 Python 的 `complex`。

        Returns:
            Union[List[complex], List[List[complex]]]:
                一个全为 `0.0 + 0.0j` 的向量（列表）或矩阵（嵌套列表）。

        Raises:
            ValueError: 如果 `shape` 中的维度值为负数，或者 `shape` 的
                        长度不是1或2。
            TypeError: 如果 `shape` 不是一个元组，或者其元素不是整数。
        """
        # --- 输入验证 ---
        if not isinstance(shape, tuple):
            self._internal_logger.error(f"zeros: 'shape' must be a tuple, but got {type(shape).__name__}.")
            raise TypeError("Shape must be a tuple.")
        
        if len(shape) == 1:
            # --- 创建一维向量 ---
            dim = shape[0]
            if not isinstance(dim, int):
                self._internal_logger.error(f"zeros: Dimension in 1D 'shape' must be an integer, but got {type(dim).__name__}.")
                raise TypeError("Dimension in shape must be an integer.")
            if dim < 0:
                self._internal_logger.error(f"zeros: Dimension in 1D 'shape' cannot be negative, but got {dim}.")
                raise ValueError("Dimension in shape cannot be negative.")
            return [0.0 + 0.0j for _ in range(dim)]
            
        elif len(shape) == 2:
            # --- 创建二维矩阵 ---
            rows, cols = shape
            if not isinstance(rows, int) or not isinstance(cols, int):
                self._internal_logger.error(f"zeros: Dimensions in 2D 'shape' must be integers, but got ({type(rows).__name__}, {type(cols).__name__}).")
                raise TypeError("Dimensions in shape must be integers.")
            if rows < 0 or cols < 0:
                self._internal_logger.error(f"zeros: Dimensions in 2D 'shape' cannot be negative, but got ({rows}, {cols}).")
                raise ValueError("Dimensions in shape cannot be negative.")
            return self.create_matrix(rows, cols, value=0.0 + 0.0j)
            
        else:
            self._internal_logger.error(f"zeros: Only 1D or 2D shapes are supported, but got shape with {len(shape)} dimensions.")
            raise ValueError(f"PurePythonBackend only supports 1D or 2D shapes for zeros, but got shape with {len(shape)} dimensions.")

    def zeros_like(self, arr: Union[List[Any], List[List[Any]]]) -> Union[List[complex], List[List[complex]]]:
        """
        创建一个与给定数组 `arr` 形状相同且类型为 `complex` 的全零数组。

        Args:
            arr (Union[List[Any], List[List[Any]]]): 
                输入数组，可以是列表或嵌套列表。

        Returns:
            Union[List[complex], List[List[complex]]]:
                一个全为 `0.0 + 0.0j` 且与 `arr` 形状相同的数组。

        Raises:
            TypeError: 如果 `arr` 不是列表或嵌套列表。
            ValueError: 如果 `arr` 是空列表或形状不规整。
        """
        if not isinstance(arr, list):
            self._internal_logger.error(f"zeros_like: Input 'arr' must be a list or nested list, but got {type(arr).__name__}.")
            raise TypeError("Input 'arr' must be a list or nested list.")
        
        if not arr: # 空列表
            return []
        
        if isinstance(arr[0], list): # 二维矩阵
            rows = len(arr)
            cols = len(arr[0])
            for r in range(rows):
                if not isinstance(arr[r], list) or len(arr[r]) != cols:
                    self._internal_logger.error(f"zeros_like: Input 'arr' is not a well-formed matrix (rows have different lengths).")
                    raise ValueError("Input 'arr' is not a well-formed matrix (rows have different lengths).")
            return self.create_matrix(rows, cols, value=0.0 + 0.0j)
        else: # 一维向量
            dim = len(arr)
            return [0.0 + 0.0j for _ in range(dim)]

    # --- 核心线性代数操作 ---

    def dot(self, mat_a: List[List[complex]], mat_b: Union[List[List[complex]], List[complex]]) -> Union[List[List[complex]], List[complex]]:
        """
        [矩阵-向量乘法修正版] 计算矩阵乘法 (dot product)，C = A @ B。
        
        此版本经过全面重构，能够正确处理两种核心的乘法操作：
        1. 矩阵-矩阵乘法：当 `mat_b` 是一个二维列表（矩阵）时。
        2. 矩阵-向量乘法：当 `mat_b` 是一个一维列表（向量）时。

        它通过检查 `mat_b` 的结构来自动区分这两种情况，并分派到相应的、
        经过验证的计算逻辑。此修复对于 `PurePythonBackend` 在态矢量模拟中
        正确应用全局酉算子至关重要。

        Args:
            mat_a (List[List[complex]]):
                乘法中的左矩阵 (A)。必须是一个非空的、形状规整的嵌套列表。
            mat_b (Union[List[List[complex]], List[complex]]):
                乘法中的右矩阵 (B) 或右向量 (b)。可以是嵌套列表或一维列表。

        Returns:
            Union[List[List[complex]], List[complex]]:
                如果 mat_b 是矩阵，则返回结果矩阵 C（二维列表）。
                如果 mat_b 是向量，则返回结果向量 c（一维列表）。

        Raises:
            TypeError: 如果输入格式不正确（例如，不是列表，或元素非数值）。
            ValueError: 如果矩阵或向量的维度不满足乘法要求，或形状不规整。
        """
        # --- 步骤 1: 严格验证左矩阵 A ---
        if not isinstance(mat_a, list) or not mat_a or not isinstance(mat_a[0], list):
            self._internal_logger.error(f"dot: Input 'mat_a' must be a non-empty nested list (matrix). Got {type(mat_a).__name__}.")
            raise TypeError("Input 'mat_a' must be a non-empty nested list (matrix).")
        
        rows_a = len(mat_a)
        cols_a = len(mat_a[0])
        if cols_a == 0:
            self._internal_logger.error(f"dot: Input 'mat_a' cannot have rows of zero length.")
            raise ValueError("Input 'mat_a' cannot have rows of zero length.")
        for r in range(rows_a):
            if not isinstance(mat_a[r], list) or len(mat_a[r]) != cols_a:
                self._internal_logger.error(f"dot: Input 'mat_a' is not a well-formed matrix (rows have different lengths).")
                raise ValueError("Input 'mat_a' is not a well-formed matrix.")

        # --- 步骤 2: 判断 mat_b 是矩阵还是向量，并分派到不同路径 ---
        
        # 通过检查第一个元素是否是列表来区分。空列表也被视为向量。
        is_vec_b = False
        if isinstance(mat_b, list):
            if not mat_b or not isinstance(mat_b[0], list):
                is_vec_b = True
        else:
            self._internal_logger.error(f"dot: Input 'mat_b' must be a list or a nested list, but got {type(mat_b).__name__}.")
            raise TypeError(f"Input 'mat_b' must be a list or a nested list.")

        if is_vec_b:
            # --- 路径 A: 矩阵-向量乘法 (A @ b = c) ---
            vec_b = mat_b
            len_b = len(vec_b)

            # 验证维度兼容性 (A的列数必须等于b的长度)
            if cols_a != len_b:
                self._internal_logger.error(
                    f"dot: Dimensions are not compatible for matrix-vector product. "
                    f"Matrix A has shape ({rows_a}x{cols_a}) and vector b has length {len_b}. "
                    f"Inner dimensions ({cols_a} and {len_b}) must match."
                )
                raise ValueError(
                    f"Dimensions are not compatible for matrix-vector product: "
                    f"Matrix A ({rows_a}x{cols_a}) and vector b (length={len_b})."
                )

            # --- 计算 ---
            # 初始化一个长度为 `rows_a` 的结果向量
            res_vec = [0.0 + 0.0j] * rows_a
            for i in range(rows_a):
                # 计算结果向量的第 i 个元素，即 A 的第 i 行与向量 b 的点积
                sum_val = 0.0 + 0.0j
                for k in range(cols_a):
                    sum_val += complex(mat_a[i][k]) * complex(vec_b[k])
                res_vec[i] = sum_val
            return res_vec

        else:
            # --- 路径 B: 矩阵-矩阵乘法 (A @ B = C) ---
            # 此时我们已经知道 mat_b 是一个非空的嵌套列表，进行进一步验证
            rows_b = len(mat_b)
            cols_b = len(mat_b[0])
            for r in range(rows_b):
                if not isinstance(mat_b[r], list) or len(mat_b[r]) != cols_b:
                    self._internal_logger.error(f"dot: Input 'mat_b' is not a well-formed matrix (rows have different lengths).")
                    raise ValueError("Input 'mat_b' is not a well-formed matrix.")
            
            # 验证维度兼容性 (A的列数必须等于B的行数)
            if cols_a != rows_b:
                self._internal_logger.error(
                    f"dot: Matrix dimensions are not compatible for dot product. "
                    f"Matrix A has shape ({rows_a}x{cols_a}) and Matrix B has shape ({rows_b}x{cols_b}). "
                    f"Inner dimensions ({cols_a} and {rows_b}) must match."
                )
                raise ValueError(
                    f"Matrix dimensions are not compatible for dot product: "
                    f"A ({rows_a}x{cols_a}) and B ({rows_b}x{cols_b})."
                )

            # --- 计算 ---
            # 初始化一个 (rows_a x cols_b) 的结果矩阵
            res_mat = self.create_matrix(rows_a, cols_b)
            for i in range(rows_a):
                for j in range(cols_b):
                    sum_val = 0.0 + 0.0j
                    for k in range(cols_a): # 或者 range(rows_b)
                        sum_val += complex(mat_a[i][k]) * complex(mat_b[k][j])
                    res_mat[i][j] = sum_val
            return res_mat
    def kron(self, mat_a: List[List[complex]], mat_b: List[List[complex]]) -> List[List[complex]]:
        """
        计算两个矩阵的张量积（Kronecker product），C = A ⊗ B。

        此函数严格按照张量积的数学定义，通过四个嵌套循环来实现。
        它不依赖任何外部库，用于替代 `numpy.kron` 或 `cupy.kron`。

        结果矩阵 C 是一个分块矩阵，其中第 (i,j) 个块是 `A[i,j] * B`。
        结果矩阵的维度是 `(rows_a * rows_b, cols_a * cols_b)`。

        **比特序约定 (Endianness Convention):**
        在量子计算中，`kron(A, B)` 通常对应于算子 `A ⊗ B`。如果我们将
        希尔伯特空间表示为 `H_A ⊗ H_B`，其中 `H_A` 是第一个（更高位）
        子空间，`H_B` 是第二个（更低位）子空间，那么这个函数的结果
        与 `A` 作用于高位比特、`B` 作用于低位比特的约定是一致的。
        例如，对于 `|q_1 q_0>` 基矢，`kron(Op_1, Op_0)` 会生成正确的全局算子。

        Args:
            mat_a (List[List[complex]]):
                张量积中的左矩阵 (A)。必须是一个非空的、形状规整的嵌套列表。
            mat_b (List[List[complex]]):
                张量积中的右矩阵 (B)。必须是一个非空的、形状规整的嵌套列表。

        Returns:
            List[List[complex]]:
                结果矩阵 C。

        Raises:
            TypeError: 如果输入 `mat_a` 或 `mat_b` 不是嵌套列表。
            ValueError: 如果任一矩阵为空或形状不规整（各行长度不一）。
        """
        # --- 步骤 1: 严格的输入验证 ---

        def _validate_matrix_for_kron(mat, name):
            if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
                self._internal_logger.error(f"kron: Input '{name}' must be a non-empty nested list (matrix). Got {type(mat).__name__}.")
                raise TypeError(f"Input '{name}' must be a non-empty nested list (matrix).")
            rows = len(mat)
            cols = len(mat[0])
            if cols == 0:
                self._internal_logger.error(f"kron: Input '{name}' cannot have rows of zero length. Shape: ({rows}x{cols}).")
                raise ValueError(f"Input '{name}' cannot have rows of zero length.")
            for r in range(rows):
                if not isinstance(mat[r], list) or len(mat[r]) != cols:
                    self._internal_logger.error(f"kron: Input '{name}' is not a well-formed matrix (rows have different lengths). Row {r} has length {len(mat[r])}, expected {cols}.")
                    raise ValueError(f"Input '{name}' is not a well-formed matrix (rows have different lengths).")
            return rows, cols

        rows_a, cols_a = _validate_matrix_for_kron(mat_a, 'mat_a')
        rows_b, cols_b = _validate_matrix_for_kron(mat_b, 'mat_b')

        # --- 步骤 2: 核心实现 ---

        # 计算结果矩阵的维度
        res_rows = rows_a * rows_b
        res_cols = cols_a * cols_b
        
        # 创建一个正确大小的全零结果矩阵
        res = self.create_matrix(res_rows, res_cols)
        
        for i_a in range(rows_a):
            for j_a in range(cols_a):
                # 获取 A 矩阵的标量元素
                scalar_a = complex(mat_a[i_a][j_a])
                
                # 如果 A 的元素是 0，则对应的整个 B 块都是 0，可以跳过以优化
                if self.isclose(scalar_a, 0.0 + 0.0j):
                    continue

                for i_b in range(rows_b):
                    for j_b in range(cols_b):
                        # 获取 B 矩阵的标量元素
                        scalar_b = complex(mat_b[i_b][j_b])
                        
                        # 计算结果矩阵中的全局行索引和列索引
                        global_row = i_a * rows_b + i_b
                        global_col = j_a * cols_b + j_b
                        
                        res[global_row][global_col] = scalar_a * scalar_b
                        
        return res

    def outer(self, vec_a: List[complex], vec_b: List[complex]) -> List[List[complex]]:
        """
        计算两个向量的外积 (outer product)，结果为一个矩阵 M = |a⟩⟨b*|。

        外积的结果矩阵 M 的元素由 `M[i][j] = a[i] * b[j]` 定义。
        在量子计算中，这常用于从一个态矢量 `|ψ⟩` 构建其对应的密度矩阵 `ρ`，
        通过计算 `|ψ⟩` 与其共轭转置 `⟨ψ|` 的外积，即 `ρ = |ψ⟩⟨ψ|`。
        为了计算 `ρ`，调用时 `vec_b` 应该是 `vec_a` 的逐元素共轭。

        此函数用于替代 `numpy.outer`。

        Args:
            vec_a (List[complex]):
                左向量 `|a⟩` (列向量)。必须是一个一维列表。
            vec_b (List[complex]):
                右向量 `⟨b|` (行向量)。必须是一个一维列表。

        Returns:
            List[List[complex]]:
                结果矩阵 M，其形状为 `(len(vec_a), len(vec_b))`。

        Raises:
            TypeError: 如果输入 `vec_a` 或 `vec_b` 不是一维列表。
            ValueError: 如果任一向量为空，或包含非数值元素。
        """
        # --- 步骤 1: 严格的输入验证 ---

        def _validate_vector(vec, name):
            if not isinstance(vec, list):
                self._internal_logger.error(f"outer: Input '{name}' must be a 1D list (vector). Got {type(vec).__name__}.")
                raise TypeError(f"Input '{name}' must be a 1D list (vector).")
            if not vec:
                self._internal_logger.warning(f"outer: Input '{name}' is an empty vector. This will result in an empty or zero matrix.")
                return 0
            return len(vec)

        len_a = _validate_vector(vec_a, 'vec_a')
        len_b = _validate_vector(vec_b, 'vec_b')
            
        # --- 步骤 2: 核心实现 ---

        # 处理空向量情况
        if len_a == 0 or len_b == 0:
            return [] if len_a == 0 else self.create_matrix(len_a, 0) # 遵循 NumPy outer 的行为

        # 创建一个正确大小的全零结果矩阵
        mat = self.create_matrix(len_a, len_b)
        
        for i in range(len_a):
            val_a = complex(vec_a[i])
            if self.isclose(val_a, 0.0 + 0.0j): # 优化：如果 a[i] 是 0，则整行都是 0
                continue

            for j in range(len_b):
                val_b = complex(vec_b[j])
                mat[i][j] = val_a * val_b
                
        return mat

    def conj_transpose(self, mat: List[List[complex]]) -> List[List[complex]]:
        """
        计算矩阵的共轭转置 (conjugate transpose)，也称为厄米共轭或 Dagger (†)。

        共轭转置包含两个步骤：
        1.  **转置 (Transpose)**: 交换矩阵的行和列 (`M_T[j][i] = M[i][j]`)。
        2.  **共轭 (Conjugate)**: 取矩阵中每个元素的复共轭 (`a + bi` -> `a - bi`)。

        在量子力学中，此操作至关重要：
        -   将一个列向量表示的态矢量 `|ψ⟩` 转换为其对偶的行向量 `⟨ψ|`。
        -   对于酉矩阵 `U`，其逆 `U⁻¹` 等于其共轭转置 `U†`。
        -   可观测量（厄米算子）满足 `H = H†`。

        此函数用于替代 NumPy 中的 `.conj().T` 操作。

        Args:
            mat (List[List[complex]]):
                输入矩阵 M。必须是一个非空的、形状规整的嵌套列表。

        Returns:
            List[List[complex]]:
                结果矩阵 M†，其形状为 `(cols, rows)`。

        Raises:
            TypeError: 如果输入 `mat` 不是嵌套列表。
            ValueError: 如果矩阵为空或形状不规整（各行长度不一）。
        """
        # --- 步骤 1: 严格的输入验证 ---

        if not isinstance(mat, list):
            self._internal_logger.error(f"conj_transpose: Input 'mat' must be a list of lists (matrix). Got {type(mat).__name__}.")
            raise TypeError("Input 'mat' must be a list of lists (matrix).")
        
        if not mat: # 处理空矩阵的边缘情况
            return []

        if not isinstance(mat[0], list):
             self._internal_logger.error(f"conj_transpose: Input 'mat' must be a nested list (matrix). First element is {type(mat[0]).__name__}.")
             raise TypeError("Input 'mat' must be a nested list (matrix).")

        rows = len(mat)
        cols = len(mat[0])
        if cols == 0:
            # 如果是 [[], [], []] 这种，则返回 [] 是合理的
            self._internal_logger.warning(f"conj_transpose: Input matrix has rows of zero length. Shape: ({rows}x{cols}). Returning empty list.")
            return [] 

        for r in range(rows):
            if not isinstance(mat[r], list) or len(mat[r]) != cols:
                self._internal_logger.error(f"conj_transpose: Input matrix is not a well-formed matrix (rows have different lengths). Row {r} has length {len(mat[r])}, expected {cols}.")
                raise ValueError(f"Input '{name}' is not a well-formed matrix (rows have different lengths).")

        # --- 步骤 2: 核心实现 ---

        # 创建一个转置后大小的结果矩阵 (cols x rows)
        res = self.create_matrix(cols, rows)
        
        for i in range(rows):
            for j in range(cols):
                res[j][i] = complex(mat[i][j]).conjugate()
                
        return res

    def trace(self, mat: List[List[complex]]) -> complex:
        """
        计算一个方阵的迹 (trace)。

        矩阵的迹是其主对角线上所有元素的总和。
        `Tr(M) = Σ_i M[i,i]`

        在量子力学中，迹具有至关重要的意义：
        1.  **概率守恒**: 一个有效的密度矩阵 `ρ` 的迹必须为1 (`Tr(ρ) = 1`)。
            此函数用于验证量子态是否被正确归一化。
        2.  **期望值计算**: 一个可观测量 `O` 在量子态 `ρ` 下的期望值
            由公式 `⟨O⟩ = Tr(O @ ρ)` 给出。

        此函数用于替代 `numpy.trace`。

        Args:
            mat (List[List[complex]]):
                输入矩阵 M。必须是一个非空的、形状规整的方阵（行数==列数）。

        Returns:
            complex:
                矩阵主对角线元素的和。

        Raises:
            TypeError: 如果输入 `mat` 不是嵌套列表。
            ValueError: 如果矩阵不是方阵，或者为空、形状不规整。
        """
        # --- 步骤 1: 严格的输入验证 ---

        if not isinstance(mat, list):
            self._internal_logger.error(f"trace: Input 'mat' must be a list of lists (matrix). Got {type(mat).__name__}.")
            raise TypeError("Input 'mat' must be a list of lists (matrix).")
        
        if not mat:  # 处理空矩阵的边缘情况
            self._internal_logger.warning("trace: Input matrix is empty. Trace is defined as 0.0 + 0.0j.")
            return 0.0 + 0.0j 

        if not isinstance(mat[0], list):
             self._internal_logger.error(f"trace: Input 'mat' must be a nested list (matrix). First element is {type(mat[0]).__name__}.")
             raise TypeError("Input 'mat' must be a nested list (matrix).")

        rows = len(mat)
        cols = len(mat[0])
        if cols == 0: # 如果是 [[], [], []] 这种
            self._internal_logger.warning("trace: Input matrix has rows of zero length. Trace is defined as 0.0 + 0.0j.")
            return 0.0 + 0.0j 
             
        if rows != cols:
            self._internal_logger.error(f"trace: Trace is only defined for square matrices, but got a matrix of shape ({rows}x{cols}).")
            raise ValueError(f"Trace is only defined for square matrices, but got a matrix of shape ({rows}x{cols}).")
        
        for r in range(rows):
            if not isinstance(mat[r], list) or len(mat[r]) != cols:
                self._internal_logger.error(f"trace: Input matrix is not a well-formed matrix (rows have different lengths). Row {r} has length {len(mat[r])}, expected {cols}.")
                raise ValueError(f"Input '{name}' is not a well-formed matrix (rows have different lengths).")

        # --- 2. 核心实现 ---

        trace_sum = 0.0 + 0.0j
        for i in range(rows):
            trace_sum += complex(mat[i][i])
            
        return trace_sum

    def diag(self, mat: List[List[complex]]) -> List[complex]:
        """
        提取一个方阵的主对角线元素，并返回一个向量。

        `diag(M)` 返回一个列表，其中第 `i` 个元素是 `M[i,i]`。

        在量子计算中，此函数的核心应用是：
        -   从一个密度矩阵 `ρ` 中提取所有计算基的测量概率。
            对于一个处于混合态或纯态的量子系统，测量到计算基矢 `|i⟩`
            的概率由密度矩阵的第 `i` 个对角线元素 `ρ[i,i]` 给出。

        此函数用于替代 `numpy.diag`（当用于提取对角线时）。

        Args:
            mat (List[List[complex]]):
                输入矩阵 M。必须是一个非空的、形状规整的方阵（行数==列数）。

        Returns:
            List[complex]:
                一个包含矩阵主对角线元素的一维列表（向量）。

        Raises:
            TypeError: 如果输入 `mat` 不是嵌套列表。
            ValueError: 如果矩阵不是方阵，或者为空、形状不规整。
        """
        # --- 步骤 1: 严格的输入验证 ---

        if not isinstance(mat, list):
            self._internal_logger.error(f"diag: Input 'mat' must be a list of lists (matrix). Got {type(mat).__name__}.")
            raise TypeError("Input 'mat' must be a list of lists (matrix).")
        
        if not mat:  # 处理空矩阵的边缘情况
            self._internal_logger.warning("diag: Input matrix is empty. Diagonal is an empty list.")
            return []

        if not isinstance(mat[0], list):
             self._internal_logger.error(f"diag: Input 'mat' must be a nested list (matrix). First element is {type(mat[0]).__name__}.")
             raise TypeError("Input 'mat' must be a nested list (matrix).")
        
        rows = len(mat)
        cols = len(mat[0])
        if cols == 0: # 如果是 [[], [], []] 这种
            self._internal_logger.warning("diag: Input matrix has rows of zero length. Diagonal is an empty list.")
            return []
        
        if rows != cols:
            self._internal_logger.error(f"diag: Diagonal can only be extracted from a square matrix in this context, but got shape ({rows}x{cols}).")
            raise ValueError(f"Diagonal can only be extracted from a square matrix in this context, but got shape ({rows}x{cols}).")
        
        for r in range(rows):
            if not isinstance(mat[r], list) or len(mat[r]) != cols:
                self._internal_logger.error(f"diag: Input matrix is not a well-formed matrix (rows have different lengths). Row {r} has length {len(mat[r])}, expected {cols}.")
                raise ValueError(f"Input '{name}' is not a well-formed matrix (rows have different lengths).")

        # --- 2. 核心实现 ---

        return [complex(mat[i][i]) for i in range(rows)]

    @staticmethod
    def _static_flatten(nested_list: Any) -> List[complex]:
        """
        [健壮性改进版] 一个静态辅助方法，使用非递归的栈实现，将任意嵌套
        的列表扁平化为一维复数列表。

        Args:
            nested_list (Any): 任意嵌套深度的列表。

        Returns:
            List[complex]: 扁平化后的一维复数列表。
        """
        flat_list: List[complex] = []
        stack = [iter([nested_list])]

        while stack:
            try:
                element = next(stack[-1])
                if isinstance(element, list):
                    stack.append(iter(element))
                else:
                    try:
                        flat_list.append(complex(element))
                    except (TypeError, ValueError) as e:
                        logging.getLogger(__name__).warning(f"_static_flatten: Could not convert element '{element}' (type: {type(element).__name__}) to complex. Skipping. Error: {e}")
            except StopIteration:
                stack.pop()
        
        return flat_list

    @staticmethod
    def _build_from_iterator(shape_tuple: Tuple[int, ...], data_iterator: Any) -> Any:
        """
        [健壮性改进版] 一个静态辅助方法，通过消耗一个迭代器来递归地构建嵌套列表。

        Args:
            shape_tuple (Tuple[int, ...]): 目标形状的元组。
            data_iterator (Iterator[complex]): 包含扁平化数据的迭代器。

        Returns:
            Any: 递归构建的嵌套列表。

        Raises:
            StopIteration: 如果数据迭代器在构建过程中耗尽。
            ValueError: 如果形状元组或其元素无效。
        """
        if not isinstance(shape_tuple, tuple):
            logging.getLogger(__name__).error(f"_build_from_iterator: 'shape_tuple' must be a tuple, but got {type(shape_tuple).__name__}.")
            raise TypeError("'shape_tuple' must be a tuple.")
        
        if not shape_tuple:
            try:
                return next(data_iterator)
            except StopIteration:
                logging.getLogger(__name__).error("_build_from_iterator: Data iterator exhausted when expecting a scalar element for empty shape.")
                raise ValueError("Data iterator exhausted for scalar element.")
        
        current_dim_size = shape_tuple[0]
        if not isinstance(current_dim_size, int) or current_dim_size < 0:
            logging.getLogger(__name__).error(f"_build_from_iterator: Dimension size must be a non-negative integer, but got {current_dim_size}.")
            raise ValueError("Dimension size must be a non-negative integer.")

        deeper_shape = shape_tuple[1:]
        result_list = []
        for _ in range(current_dim_size):
            result_list.append(PurePythonBackend._build_from_iterator(deeper_shape, data_iterator))
        return result_list
    
    def reshape(self, data: Any, new_shape: Tuple[int, ...]) -> Any:
        """
        [健壮性改进版] 将一个向量、矩阵或任意嵌套的列表重塑为新的指定形状。

        此版本支持 `-1` 推断和高效的迭代器构建，并对边缘情况处理得更加明确。

        Args:
            data (Any): 输入的数据，可以是任意嵌套深度的列表或一维列表。
            new_shape (Tuple[int, ...]): 一个整数元组，指定了输出张量的新维度。

        Returns:
            Any: 重塑后的、代表多维张量的嵌套列表。

        Raises:
            TypeError: 如果 `new_shape` 不是元组或其元素不是整数。
            ValueError: 如果 `new_shape` 包含多个 `-1`，或者数据大小与目标形状不兼容。
        """
        if not isinstance(new_shape, tuple):
            self._internal_logger.error(f"reshape: 'new_shape' must be a tuple of integers, but got {type(new_shape).__name__}.")
            raise TypeError("new_shape must be a tuple of integers.")

        # --- 步骤 1: 扁平化 ---
        flat_list = self._static_flatten(data)
        current_size = len(flat_list)
        
        # --- 步骤 2: 维度推断与验证 ---
        final_shape_list = list(new_shape)
        num_neg_one = final_shape_list.count(-1)

        if num_neg_one > 1:
            self._internal_logger.error(f"reshape: 'new_shape' {new_shape} can only contain one '-1', but found {num_neg_one}.")
            raise ValueError(f"Cannot reshape: new_shape {new_shape} can only contain one '-1'.")
        
        if num_neg_one == 1:
            product_of_known_dims = 1
            for dim in final_shape_list:
                if dim > 0:
                    product_of_known_dims *= dim
                elif dim == 0:
                    if current_size != 0:
                         self._internal_logger.error(f"reshape: Cannot infer '-1' in shape {new_shape} for data of size {current_size} when a zero dimension is present (product_of_known_dims=0).")
                         raise ValueError(f"Cannot infer '-1' in shape {new_shape} for data of size {current_size} when a zero dimension is present.")
                    inferred_dim = 0 
                    break 
            else: # 如果没有 0 维度
                if product_of_known_dims == 0: 
                    inferred_dim = current_size 
                elif current_size % product_of_known_dims != 0:
                    self._internal_logger.error(f"reshape: Cannot reshape array of size {current_size} into shape {new_shape} (product of known dims is {product_of_known_dims}).")
                    raise ValueError(f"Cannot reshape array of size {current_size} into shape {new_shape}.")
                else:
                    inferred_dim = current_size // product_of_known_dims
            
            final_shape_list[final_shape_list.index(-1)] = inferred_dim
        
        final_shape = tuple(final_shape_list)

        # --- 步骤 3: 尺寸验证 ---
        if current_size != self.math.prod(final_shape): 
            self._internal_logger.error(
                f"reshape: Data size {current_size} is incompatible with target shape {final_shape} (inferred from {new_shape}). "
                f"Product of target shape dimensions is {self.math.prod(final_shape)}."
            )
            raise ValueError(
                f"Cannot reshape: data size {current_size} is incompatible with target shape {final_shape} "
                f"(inferred from {new_shape})."
            )
            
        # --- 步骤 4: 处理边缘情况并构建 ---
        if current_size == 0:
            if not final_shape: return None 
            def _build_empty(shape_part):
                if not shape_part: return [] 
                return [self._build_empty(shape_part[1:]) for _ in range(shape_part[0])] # 递归调用实例方法
            return _build_empty(final_shape)

        if not final_shape: # 目标是标量
            if current_size != 1:
                self._internal_logger.error(f"reshape: Cannot reshape array of size {current_size} into scalar (empty shape tuple). Expected size 1.")
                raise ValueError("Cannot reshape array of size > 1 into scalar.")
            return flat_list[0]

        data_iterator = iter(flat_list)
        try:
            return self._build_from_iterator(final_shape, data_iterator)
        except StopIteration as e:
            self._internal_logger.error(f"reshape: Data iterator exhausted prematurely during building. This indicates a logic error or inconsistent size calculation. Error: {e}")
            raise ValueError("Data iterator exhausted prematurely during reshape.") from e


    @staticmethod
    def _parse_einsum_for_partial_trace(subscripts: str, num_qubits: int) -> Tuple[List[int], List[int]]:
        """
        [健壮性改进版] 解析 einsum 字符串以获取部分迹操作的量子比特索引。

        Args:
            subscripts (str): einsum 字符串，例如 "abc...ABC... -> de...DE..."。
            num_qubits (int): 系统的总量子比特数。

        Returns:
            Tuple[List[int], List[int]]: 
                一个元组 `(qubits_to_keep, qubits_to_trace)`。
                - `qubits_to_keep`: 要保留的量子比特的全局索引列表。
                - `qubits_to_trace`: 要迹掉的量子比特的全局索引列表。

        Raises:
            ValueError: 如果 `subscripts` 格式不正确，或 `num_qubits` 无效。
        """
        if not isinstance(subscripts, str) or '->' not in subscripts:
            logging.getLogger(__name__).error(f"_parse_einsum_for_partial_trace: Invalid 'subscripts' format: '{subscripts}'. Expected 'input->output'.")
            raise ValueError("Invalid 'subscripts' format. Expected 'input->output'.")
        if not isinstance(num_qubits, int) or num_qubits < 0:
            logging.getLogger(__name__).error(f"_parse_einsum_for_partial_trace: Invalid 'num_qubits': {num_qubits}. Must be a non-negative integer.")
            raise ValueError("Invalid 'num_qubits'.")

        input_subs, output_subs = subscripts.split('->')

        # 确保输入字符串的长度与 2*num_qubits 匹配（bra 和 ket 各 num_qubits）
        if len(input_subs) != 2 * num_qubits:
            logging.getLogger(__name__).error(f"_parse_einsum_for_partial_trace: Input subscript length ({len(input_subs)}) does not match 2 * num_qubits ({2*num_qubits}).")
            raise ValueError("Input subscript length does not match 2 * num_qubits.")

        bra_subs = input_subs[:num_qubits]
        ket_subs = input_subs[num_qubits:]

        qubits_to_keep: List[int] = []
        qubits_to_trace: List[int] = []

        # 约定：einsum 字符串中的索引顺序是从最高位量子比特到最低位量子比特。
        # 即，字符串的第一个字符对应 q_N-1，第二个字符对应 q_N-2，以此类推，
        # 直到最后一个字符对应 q_0。
        for i in range(num_qubits):
            # 获取当前量子比特的全局索引
            # global_qubit_idx = num_qubits - 1 - i  # 实际量子比特索引

            # 如果 bra 和 ket 的下标字符在输出中都出现，或者它们不相同，则保留。
            # 如果 bra 和 ket 的下标字符相同，且在输出中只出现一次，则迹掉。
            # 简化逻辑：迹掉的比特，其 bra 和 ket 下标字符是相同的，且这个字符不在输出中。
            # 保留的比特，其 bra 和 ket 下标字符会分别在输出中出现。
            
            # 迹掉的条件：bra_char == ket_char (且这个 char 不在 output 中)
            # 例如 "aBc,AbC->aA" (q2q1q0, Q2Q1Q0 -> q2Q2) 迹掉 q1, q0
            # 简化为 bra_char == ket_char 则迹掉，否则保留
            if bra_subs[i] == ket_subs[i]: # 相同下标意味着迹掉 (e.g., c_i, c_i)
                qubits_to_trace.append(num_qubits - 1 - i)
            else: # 不同下标意味着保留 (e.g., b_i, B_i)
                qubits_to_keep.append(num_qubits - 1 - i)
        
        # 最终检查输出下标是否匹配保留的比特数量
        expected_output_len = 2 * len(qubits_to_keep)
        if len(output_subs) != expected_output_len:
             logging.getLogger(__name__).warning(f"_parse_einsum_for_partial_trace: Output subscript length ({len(output_subs)}) does not match expected length ({expected_output_len}) for kept qubits {qubits_to_keep}.")
        
        return sorted(qubits_to_keep), sorted(qubits_to_trace)

    def einsum(self, subscripts: str, *tensors: Any) -> Any:
        """
        [健壮性改进版] PurePythonBackend 的 einsum 实现，专注于部分迹操作。

        Args:
            subscripts (str): 描述 einsum 操作的字符串。
            *tensors (Any): 一个或两个输入张量。

        Returns:
            Any: 计算结果。

        Raises:
            TypeError: 如果输入张量类型不正确。
            ValueError: 如果 `subscripts` 格式不正确或张量数量不支持。
            RuntimeError: 如果内部计算失败。
        """
        if not isinstance(subscripts, str) or '->' not in subscripts:
            self._internal_logger.error(f"einsum: Invalid 'subscripts' format: '{subscripts}'. Expected 'input->output'.")
            raise ValueError("Invalid 'subscripts' format. Expected 'input->output'.")

        if len(tensors) == 1:
            # 模式 1: 对单个密度矩阵张量进行部分迹 (e.g., "abAB->aA")
            return self._einsum_for_density_matrix(subscripts, tensors[0])
        elif len(tensors) == 2:
            # 模式 2: 对两个态矢量张量进行外积后部分迹 (e.g., "aB,Ab->aA")
            return self._einsum_for_state_vector(subscripts, tensors[0], tensors[1])
        else:
            self._internal_logger.error(f"einsum: Only 1 or 2 tensors are supported, but {len(tensors)} were given.")
            raise ValueError(f"PurePythonBackend.einsum only supports 1 or 2 tensors, but {len(tensors)} were given.")

    def _einsum_for_density_matrix(self, subscripts: str, tensor: Any) -> Any:
        """
        [健壮性改进版] 对一个密度矩阵张量执行部分迹。

        Args:
            subscripts (str): einsum 下标字符串 (e.g., "abcABC->aA")
            tensor (Any): 输入的密度矩阵张量，应是形状为 (2, 2, ..., 2) 的嵌套列表。

        Returns:
            Any: 约化后的密度矩阵。

        Raises:
            TypeError: 如果 `tensor` 类型不正确。
            ValueError: 如果 `tensor` 形状不正确或下标解析失败。
        """
        if not isinstance(tensor, list):
             self._internal_logger.error(f"_einsum_for_density_matrix: Input 'tensor' must be a nested list, but got {type(tensor).__name__}.")
             raise TypeError("Input 'tensor' must be a nested list.")
        
        # 从扁平化张量推断总比特数
        flat_tensor = self._static_flatten(tensor)
        if not flat_tensor: return [] # 空张量返回空列表
        
        num_total_qubits = round(self.math.log2(len(flat_tensor)))
        if (1 << num_total_qubits) != len(flat_tensor):
            self._internal_logger.error(f"_einsum_for_density_matrix: Flat tensor length ({len(flat_tensor)}) is not a power of 2 for num_total_qubits={num_total_qubits}.")
            raise ValueError("Input tensor for _einsum_for_density_matrix must represent a square matrix with dimensions being a power of 2.")

        # --- 步骤 1: 解析 einsum 字符串 ---
        qubits_to_keep, qubits_to_trace = self._parse_einsum_for_partial_trace(subscripts, num_total_qubits)
        
        num_qubits_kept = len(qubits_to_keep)
        dim_out = 1 << num_qubits_kept
        reduced_rho = self.create_matrix(dim_out, dim_out)

        # --- 步骤 2: 执行计算 ---
        for bra_local_idx in range(dim_out):
            for ket_local_idx in range(dim_out):
                sum_val = 0.0 + 0.0j
                
                for trace_config in range(1 << len(qubits_to_trace)):
                    global_bra_index = 0
                    global_ket_index = 0
                    
                    for local_pos_in_kept, global_q_idx in enumerate(qubits_to_keep):
                        if (bra_local_idx >> local_pos_in_kept) & 1: 
                            global_bra_index |= (1 << global_q_idx)
                        if (ket_local_idx >> local_pos_in_kept) & 1: 
                            global_ket_index |= (1 << global_q_idx)
                            
                    for local_pos_in_traced, global_q_idx in enumerate(qubits_to_trace):
                        if (trace_config >> local_pos_in_traced) & 1: 
                            trace_bit_mask = (1 << global_q_idx)
                            global_bra_index |= trace_bit_mask
                            global_ket_index |= trace_bit_mask 
                    
                    try:
                        sum_val += tensor[global_bra_index][global_ket_index]
                    except IndexError as e:
                        self._internal_logger.error(f"_einsum_for_density_matrix: IndexError accessing tensor[{global_bra_index}][{global_ket_index}]. Global indices out of bounds. Error: {e}")
                        raise RuntimeError("Tensor indexing error during partial trace calculation.") from e

                reduced_rho[bra_local_idx][ket_local_idx] = sum_val
        
        return reduced_rho
    
    def _einsum_for_state_vector(self, subscripts: str, vec_tensor: Any, vec_tensor_conj: Any) -> Any:
        """
        [健壮性改进版] 对两个态矢量张量执行部分迹，计算约化密度矩阵。

        Args:
            subscripts (str): einsum 下标字符串 (e.g., "aB,Ab->aA")
            vec_tensor (Any): 原始态矢量张量。
            vec_tensor_conj (Any): 共轭态矢量张量。

        Returns:
            Any: 约化后的密度矩阵。

        Raises:
            TypeError: 如果输入张量类型不正确。
            ValueError: 如果张量形状不正确或下标解析失败。
        """
        if not isinstance(vec_tensor, list) or not isinstance(vec_tensor_conj, list):
             self._internal_logger.error(f"_einsum_for_state_vector: Input 'vec_tensor' and 'vec_tensor_conj' must be nested lists, but got {type(vec_tensor).__name__} and {type(vec_tensor_conj).__name__}.")
             raise TypeError("Input 'vec_tensor' and 'vec_tensor_conj' must be nested lists.")

        flat_vec = self._static_flatten(vec_tensor)
        flat_vec_conj = self._static_flatten(vec_tensor_conj)
        
        if len(flat_vec) != len(flat_vec_conj):
             self._internal_logger.error(f"_einsum_for_state_vector: Flat vector lengths mismatch: {len(flat_vec)} vs {len(flat_vec_conj)}.")
             raise ValueError("Flat vector lengths for 'vec_tensor' and 'vec_tensor_conj' must match.")
        if not flat_vec: return [] 

        num_total_qubits = round(self.math.log2(len(flat_vec)))
        if (1 << num_total_qubits) != len(flat_vec):
            self._internal_logger.error(f"_einsum_for_state_vector: Flat vector length ({len(flat_vec)}) is not a power of 2 for num_total_qubits={num_total_qubits}.")
            raise ValueError("Flat vector length must be a power of 2.")

        input_parts = subscripts.split('->')[0].split(',')
        if len(input_parts) != 2:
            self._internal_logger.error(f"_einsum_for_state_vector: Input subscripts must have exactly two parts separated by ',', but got '{input_parts}'.")
            raise ValueError("Input subscripts must have exactly two parts separated by ','.")
        
        bra_in_str = input_parts[0]
        ket_in_str = input_parts[1]
        
        ket_in_str_for_parsing = ""
        char_map: Dict[str, str] = {}
        counter = 0
        for char in ket_in_str:
            if char.isupper():
                ket_in_str_for_parsing += char
            else:
                if char not in char_map:
                    while True:
                        new_char = chr(ord('A') + counter)
                        if new_char not in bra_in_str:
                            char_map[char] = new_char
                            break
                        counter += 1
                    ket_in_str_for_parsing += char_map[char]
                else:
                    ket_in_str_for_parsing += char_map[char]
        
        combined_input_subs = bra_in_str + ket_in_str_for_parsing
        output_subs = subscripts.split('->')[1]

        einsum_str_for_parsing = f"{combined_input_subs}->{output_subs}"
        
        qubits_to_keep, qubits_to_trace = self._parse_einsum_for_partial_trace(einsum_str_for_parsing, num_total_qubits)

        num_qubits_kept = len(qubits_to_keep)
        dim_out = 1 << num_qubits_kept
        reduced_rho_matrix = self.create_matrix(dim_out, dim_out)

        for i_row_local in range(dim_out):
            for i_col_local in range(dim_out):
                sum_val = 0.0 + 0.0j
                
                for trace_config in range(1 << len(qubits_to_trace)):
                    global_bra_index = 0
                    global_ket_index = 0
                    
                    for local_idx, global_q_idx in enumerate(qubits_to_keep):
                        if (i_row_local >> local_idx) & 1:
                            global_bra_index |= (1 << global_q_idx)
                        if (i_col_local >> local_idx) & 1:
                            global_ket_index |= (1 << global_q_idx)

                    for local_idx, global_q_idx in enumerate(qubits_to_trace):
                        if (trace_config >> local_idx) & 1:
                            trace_mask = (1 << global_q_idx)
                            global_bra_index |= trace_mask
                            global_ket_index |= trace_mask
                    
                    try:
                        sum_val += flat_vec[global_bra_index] * flat_vec_conj[global_ket_index]
                    except IndexError as e:
                        self._internal_logger.error(f"_einsum_for_state_vector: IndexError accessing flat_vec[{global_bra_index}] or flat_vec_conj[{global_ket_index}]. Error: {e}")
                        raise RuntimeError("Vector indexing error during partial trace calculation.") from e

                reduced_rho_matrix[i_row_local][i_col_local] = sum_val

        return reduced_rho_matrix


    def transpose(self, mat: List[List[complex]]) -> List[List[complex]]:
        """
        计算一个矩阵的转置 (transpose)。

        Args:
            mat (List[List[complex]]):
                输入矩阵 M。必须是一个非空的、形状规整的嵌套列表。

        Returns:
            List[List[complex]]:
                转置后的新矩阵 M_T。

        Raises:
            TypeError: 如果输入 `mat` 不是嵌套列表。
            ValueError: 如果矩阵为空或形状不规整（各行长度不一）。
        """
        # --- 步骤 1: 严格的输入验证 ---

        if not isinstance(mat, list):
            self._internal_logger.error(f"transpose: Input 'mat' must be a list of lists (matrix). Got {type(mat).__name__}.")
            raise TypeError("Input 'mat' must be a list of lists (matrix).")
        
        if not mat:  # 处理空矩阵的边缘情况
            return []

        if not isinstance(mat[0], list):
             self._internal_logger.error(f"transpose: Input 'mat' must be a nested list (matrix). First element is {type(mat[0]).__name__}.")
             raise TypeError("Input 'mat' must be a nested list (matrix).")

        rows = len(mat)
        cols = len(mat[0])
        if cols == 0 and rows > 0:
            self._internal_logger.warning("transpose: Input matrix has rows of zero length. Returning empty list.")
            return []
             
        for r in range(rows):
            if not isinstance(mat[r], list) or len(mat[r]) != cols:
                self._internal_logger.error(f"transpose: Input matrix is not a well-formed matrix (rows have different lengths). Row {r} has length {len(mat[r])}, expected {cols}.")
                raise ValueError(f"Input '{name}' is not a well-formed matrix (rows have different lengths).")

        # --- 2. 核心实现 ---

        # 创建一个转置后大小 (`cols` x `rows`) 的结果矩阵
        new_rows, new_cols = cols, rows
        res = self.create_matrix(new_rows, new_cols)
        
        for i in range(rows):
            for j in range(cols):
                res[j][i] = complex(mat[i][j])
                
        return res

    # --- 浮点数比较与数学函数 ---

    def isclose(self, a: Union[complex, float, int], b: Union[complex, float, int], atol: float = 1e-9) -> bool:
        """
        判断两个数值（整数、浮点数或复数）是否在指定的绝对容差内“足够接近”。

        Args:
            a (Union[complex, float, int]): 第一个数值。
            b (Union[complex, float, int]): 第二个数值。
            atol (float, optional):
                绝对容差 (absolute tolerance)。两个数的差的绝对值
                必须小于等于这个值，才被认为是接近的。默认为 `1e-9`。

        Returns:
            bool:
                如果两个数足够接近，则返回 `True`，否则返回 `False`。

        Raises:
            TypeError: 如果 `a`, `b` 或 `atol` 的类型不正确。
        """
        # --- 输入验证 ---
        if not isinstance(a, (complex, float, int)):
            self._internal_logger.error(f"isclose: Input 'a' must be a numeric type, but got {type(a).__name__}.")
            raise TypeError(f"Input 'a' must be a numeric type, but got {type(a).__name__}.")
        if not isinstance(b, (complex, float, int)):
            self._internal_logger.error(f"isclose: Input 'b' must be a numeric type, but got {type(b).__name__}.")
            raise TypeError(f"Input 'b' must be a numeric type, but got {type(b).__name__}.")
        if not isinstance(atol, float):
            self._internal_logger.error(f"isclose: Absolute tolerance 'atol' must be a float, but got {type(atol).__name__}.")
            raise TypeError(f"Absolute tolerance 'atol' must be a float, but got {type(atol).__name__}.")
        if atol < 0:
            self._internal_logger.warning(f"isclose: 'atol' is negative ({atol}). Using its absolute value.")
            atol = abs(atol)

        return self.abs(complex(a) - complex(b)) <= atol

    def allclose(self, mat_a: Any, mat_b: Any, atol: float = 1e-9) -> bool:
        """
        判断两个矩阵或向量中的所有对应元素是否都“足够接近”。

        Args:
            mat_a (Any): 第一个矩阵或向量。
            mat_b (Any): 第二个矩阵或向量。
            atol (float, optional):
                传递给 `isclose` 的绝对容差。默认为 `1e-9`。

        Returns:
            bool:
                如果两个矩阵的所有对应元素都足够接近，则返回 `True`。
                如果形状不匹配，或者任何一对元素不够接近，则返回 `False`。

        Raises:
            TypeError: 如果 `atol` 不是浮点数。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(atol, float):
            self._internal_logger.error(f"allclose: 'atol' must be a float, got {type(atol).__name__}.")
            raise TypeError(f"Absolute tolerance 'atol' must be a float, but got {type(atol).__name__}.")
        if atol < 0:
            self._internal_logger.warning(f"allclose: 'atol' is negative ({atol}). Using its absolute value.")
            atol = abs(atol)

        flat_a = self._static_flatten(mat_a)
        flat_b = self._static_flatten(mat_b)

        if len(flat_a) != len(flat_b):
            self._internal_logger.debug(f"allclose: Sizes mismatch: {len(flat_a)} vs {len(flat_b)}. Returning False.")
            return False
        
        if not flat_a: # 如果都是空，则视为接近
            return True

        for val_a, val_b in zip(flat_a, flat_b):
            if not self.isclose(val_a, val_b, atol=atol):
                return False
        
        return True

    def sin(self, x: Union[float, int, complex]) -> Union[float, complex]:
        """
        计算一个数值的正弦值。
        """
        if isinstance(x, complex): return self.cmath.sin(x)
        if isinstance(x, (float, int)): return self.math.sin(x)
        self._internal_logger.error(f"sin: Input must be numeric, got {type(x).__name__}.")
        raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")

    def cos(self, x: Union[float, int, complex]) -> Union[float, complex]:
        """
        计算一个数值的余弦值。
        """
        if isinstance(x, complex): return self.cmath.cos(x)
        if isinstance(x, (float, int)): return self.math.cos(x)
        self._internal_logger.error(f"cos: Input must be numeric, got {type(x).__name__}.")
        raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")

    def exp(self, x: Union[float, int, complex]) -> Union[float, complex]:
        """
        计算 e (自然常数) 的 x 次幂, e^x。
        """
        if isinstance(x, complex): return self.cmath.exp(x)
        if isinstance(x, (float, int)): return self.math.exp(x)
        self._internal_logger.error(f"exp: Input must be numeric, got {type(x).__name__}.")
        raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")

    def log2(self, x: float) -> float:
        """
        计算一个实数的以2为底的对数 (log₂x)。
        """
        if not isinstance(x, (float, int)):
            self._internal_logger.error(f"log2: Input must be a real number, got {type(x).__name__}.")
            raise TypeError(f"Input for log2 must be a real number (float or int), but got {type(x).__name__}.")
        if x <= 0:
            self._internal_logger.error(f"log2: Input must be positive, got {x}.")
            raise ValueError("Input for log2 must be positive.")
        return self.math.log2(x)

    def abs(self, x: Union[float, int, complex]) -> float:
        """
        计算一个数值的绝对值（对于实数）或模（对于复数）。
        """
        if not isinstance(x, (float, int, complex)):
            self._internal_logger.error(f"abs: Input must be numeric, got {type(x).__name__}.")
            raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")
        return self.builtins.abs(x)
    def abs(self, x: Union[float, int, complex, List]) -> Union[float, List[float]]:
        """计算数值或列表中每个元素的绝对值/模。"""
        if isinstance(x, list):
            return [abs(val) for val in x]
        if not isinstance(x, (float, int, complex)):
            self._internal_logger.error(f"abs: Input must be numeric or list, got {type(x).__name__}.")
            raise TypeError(f"Input must be a numeric type or list, but got {type(x).__name__}.")
        return abs(x)
    def sqrt(self, x: Union[float, int, complex]) -> complex:
        """
        计算一个数值的平方根。
        """
        if not isinstance(x, (float, int, complex)):
            self._internal_logger.error(f"sqrt: Input must be numeric, got {type(x).__name__}.")
            raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")
        return self.cmath.sqrt(x)

    def arange(self, dim: int, dtype=None) -> List[int]:
        """
        创建一个包含从 0 到 `dim-1` 的整数序列的列表。
        """
        if not isinstance(dim, int):
            self._internal_logger.error(f"arange: Dimension (dim={dim}) must be an integer.")
            raise TypeError("Dimension (dim) for arange must be an integer.")
        if dim < 0:
            self._internal_logger.error(f"arange: Dimension (dim={dim}) for arange cannot be negative.")
            raise ValueError("Dimension (dim) for arange cannot be negative.")
        return self.builtins.list(self.builtins.range(dim))

    def power(self, base: Union[complex, float, int], exp: Union[complex, float, int]) -> Union[complex, float, int]:
        """
        计算 `base` 的 `exp` 次幂 (`base ** exp`)。
        """
        if not isinstance(base, (complex, float, int)):
            self._internal_logger.error(f"power: Input 'base' must be numeric, got {type(base).__name__}.")
            raise TypeError(f"Input 'base' must be a numeric type, but got {type(base).__name__}.")
        if not isinstance(exp, (complex, float, int)):
            self._internal_logger.error(f"power: Input 'exp' must be numeric, got {type(exp).__name__}.")
            raise TypeError(f"Input 'exp' must be a numeric type, but got {type(exp).__name__}.")
        return self.builtins.pow(base, exp)

    def sum(self, values: List[Union[complex, float, int]]) -> Union[complex, float, int]:
        """
        计算一个数值列表的总和。
        """
        if not isinstance(values, list):
            self._internal_logger.error(f"sum: Input 'values' must be a list, got {type(values).__name__}.")
            raise TypeError(f"Input must be a list of numeric values, but got {type(values).__name__}.")
        for val in values:
            if not isinstance(val, (complex, float, int)):
                self._internal_logger.warning(f"sum: Non-numeric element '{val}' encountered in list. Attempting to convert.")
        return self.builtins.sum(values, 0j)

    # --- [核心修正 1.1：为后端添加 norm_sq 方法] ---
    def norm_sq(self, vec: List[complex]) -> float:
        """计算向量的范数平方 (sum of |amp|^2)。"""
        # [健壮性改进] 对输入进行验证
        if not isinstance(vec, list):
            raise TypeError("Input 'vec' for norm_sq must be a list.")
        
        # 使用列表推导式以获得更好的性能
        return sum(amp.real**2 + amp.imag**2 for amp in vec)
    # --- 随机数生成 ---

    def _generate_single_std_normal(self) -> float:
        """
        [健壮性改进版] 使用 Box-Muller 变换生成一个标准正态分布随机数，并缓存第二个。
        """
        if self._rand_normal_cache is not None:
            z1 = self._rand_normal_cache
            self._rand_normal_cache = None
            return z1
        
        u1 = 0.0
        while u1 == 0.0:
            u1 = self.random.random()
        u2 = self.random.random()
        
        z0 = self.math.sqrt(-2.0 * self.math.log(u1)) * self.math.cos(2.0 * self.math.pi * u2)
        z1 = self.math.sqrt(-2.0 * self.math.log(u1)) * self.math.sin(2.0 * self.math.pi * u2)
        
        self._rand_normal_cache = z1
        return z0

    def choice(self, options: List[Any], p: List[float]) -> Any:
        """
        根据给定的概率分布 `p`，从 `options` 列表中随机选择一个元素。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(options, list) or not isinstance(p, list):
            self._internal_logger.error(f"choice: Inputs 'options' ({type(options).__name__}) and 'p' ({type(p).__name__}) must be lists.")
            raise TypeError("Inputs 'options' and 'p' must be lists.")
        if len(options) != len(p):
            self._internal_logger.error(f"choice: Length of 'options' ({len(options)}) and 'p' ({len(p)}) must be the same.")
            raise ValueError(f"Length of 'options' ({len(options)}) and 'p' ({len(p)}) must be the same.")
        if not options:
            self._internal_logger.warning("choice: 'options' list is empty. Cannot make a choice. Returning None.")
            return None
        if any(prob < 0 for prob in p):
            self._internal_logger.error("choice: Probabilities in 'p' cannot be negative.")
            raise ValueError("Probabilities in 'p' cannot be negative.")
        if not all(isinstance(prob, (float, int)) for prob in p):
            self._internal_logger.error("choice: All probabilities in 'p' must be numeric.")
            raise TypeError("All probabilities in 'p' must be numeric.")

        # --- 步骤 2: 概率归一化 ---
        prob_sum = self.sum(p)
        
        # [核心修复] 先取实部再转换为浮点数
        prob_sum_float = float(prob_sum.real)

        if self.isclose(prob_sum_float, 0.0, atol=1e-12):
            self._internal_logger.warning("choice: All probabilities are zero or sum to zero. Choosing uniformly random.")
            return self.random.choice(options) 
        
        normalized_p = [float(val) / prob_sum_float for val in p]

        # --- 步骤 3: 轮盘赌选择算法 ---
        r = self.random.random()
        
        cumulative_prob = 0.0
        for i, prob in enumerate(normalized_p):
            cumulative_prob += prob
            if r < cumulative_prob:
                return options[i]
                
        return options[-1]

    def random_normal(self, loc: float = 0.0, scale: float = 1.0, size: Tuple[int, ...]= (1,)) -> Union[float, List[float], List[List[float]]]:
        """
        生成服从正态（高斯）分布的随机数。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(loc, (float, int)):
            self._internal_logger.error(f"random_normal: Mean 'loc' must be a real number, got {type(loc).__name__}.")
            raise TypeError(f"Mean 'loc' must be a real number, but got {type(loc).__name__}.")
        if not isinstance(scale, (float, int)):
            self._internal_logger.error(f"random_normal: Standard deviation 'scale' must be a real number, got {type(scale).__name__}.")
            raise TypeError(f"Standard deviation 'scale' must be a real number, but got {type(scale).__name__}.")
        if scale < 0:
            self._internal_logger.error(f"random_normal: Standard deviation 'scale' cannot be negative, got {scale}.")
            raise ValueError(f"Standard deviation 'scale' cannot be negative, but got {scale}.")
        if not isinstance(size, tuple):
             self._internal_logger.error(f"random_normal: Shape 'size' must be a tuple, got {type(size).__name__}.")
             raise TypeError(f"Shape 'size' must be a tuple, but got {type(size).__name__}.")

        # --- 步骤 2: 根据 size 生成结果 ---
        if size == (1,) or size == ():
            std_normal_val = self._generate_single_std_normal()
            return std_normal_val * scale + loc
        
        elif len(size) == 1:
            n = size[0]
            if not isinstance(n, int) or n < 0:
                 self._internal_logger.error(f"random_normal: Dimension in 1D 'size' must be a non-negative integer, got {n}.")
                 raise ValueError("Dimension in 'size' must be a non-negative integer.")
            return [self._generate_single_std_normal() * scale + loc for _ in range(n)]

        elif len(size) == 2:
            rows, cols = size
            if not isinstance(rows, int) or rows < 0 or not isinstance(cols, int) or cols < 0:
                 self._internal_logger.error(f"random_normal: Dimensions in 2D 'size' must be non-negative integers, got ({rows}, {cols}).")
                 raise ValueError("Dimensions in 'size' must be non-negative integers.")
            return [[self._generate_single_std_normal() * scale + loc for _ in range(cols)] for _ in range(rows)]
        
        else:
            self._internal_logger.error(f"random_normal: Only scalar, 1D, or 2D output sizes are supported, but got {size}.")
            raise ValueError(f"PurePythonBackend random_normal only supports scalar, 1D, or 2D output sizes, but got {size}.")

    def clip(self, values: Union[List[Union[float, int]], float, int], min_val: float, max_val: float) -> Union[List[float], float]:
        """
        [健壮性与正确性增强版] 将一个数值或一个列表中的所有数值裁剪到指定的 `[min_val, max_val]` 区间内。

        此方法旨在提供一个与 NumPy/CuPy 中 `clip` 函数行为一致的纯Python实现。它能够处理
        单个标量输入或一个数值列表，并确保所有输出值都严格落在指定的边界内。

        核心增强功能:
        - 修复了因错误调用局部辅助函数而导致的 `AttributeError`。
        - 对所有输入参数的类型和值范围进行严格验证。
        - 对列表中的每个元素进行安全的类型检查和转换，如果某个元素无法转换，
          会记录错误并使用一个安全的回退值，而不是让整个函数崩溃。
        - 提供了详尽的文档和内部注释，以提高代码的可读性和可维护性。

        Args:
            values (Union[List[Union[float, int]], float, int]):
                要进行裁剪的输入。可以是一个数值（`int` 或 `float`），
                或一个包含数值的列表。
            min_val (float):
                裁剪区间的下界。所有小于此值的输入将被替换为 `min_val`。
            max_val (float):
                裁剪区间的上界。所有大于此值的输入将被替换为 `max_val`。

        Returns:
            Union[List[float], float]:
                如果输入是标量，则返回裁剪后的单个 `float` 值。
                如果输入是列表，则返回一个包含裁剪后 `float` 值的新列表。

        Raises:
            TypeError: 如果 `values` 的类型不是 `list`, `float`, 或 `int`，
                       或者 `min_val`/`max_val` 不是数值类型。
            ValueError: 如果 `min_val` 大于 `max_val`。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(values, (list, float, int)):
            self._internal_logger.error(f"clip: Input 'values' must be a list, float, or int, but got {type(values).__name__}.")
            raise TypeError(f"Input 'values' must be a list, float, or int, but got {type(values).__name__}.")
        if not isinstance(min_val, (float, int)):
            self._internal_logger.error(f"clip: Boundary 'min_val' must be a numeric type, but got {type(min_val).__name__}.")
            raise TypeError(f"Boundary 'min_val' must be a numeric type, but got {type(min_val).__name__}.")
        if not isinstance(max_val, (float, int)):
            self._internal_logger.error(f"clip: Boundary 'max_val' must be a numeric type, but got {type(max_val).__name__}.")
            raise TypeError(f"Boundary 'max_val' must be a numeric type, but got {type(max_val).__name__}.")
        
        # 确保边界值逻辑正确
        if min_val > max_val:
            self._internal_logger.error(f"clip: The minimum value ({min_val}) cannot be greater than the maximum value ({max_val}).")
            raise ValueError(f"The minimum value ({min_val}) cannot be greater than the maximum value ({max_val}).")

        # 将边界值预先转换为浮点数，以避免在循环中重复转换
        f_min_val = float(min_val)
        f_max_val = float(max_val)

        # --- 步骤 2: 定义一个局部的、健壮的辅助函数来处理单个值的裁剪 ---
        def _clip_single(value):
            """一个局部的辅助函数，负责安全地转换和裁剪单个值。"""
            # 检查值是否为数值类型
            if not isinstance(value, (float, int)):
                self._internal_logger.warning(f"clip: Non-numeric value '{value}' (type: {type(value).__name__}) encountered. Attempting conversion to float.")
                try:
                    # 尝试将非数值类型转换为浮点数
                    value = float(value)
                except (TypeError, ValueError):
                    # 如果转换失败，记录一个错误并使用一个安全的回退值
                    self._internal_logger.error(f"clip: Failed to convert non-numeric value '{value}' to float for clipping. Using min_val ({f_min_val}) as fallback.")
                    return f_min_val
            
            # 使用内置的 max/min 函数执行裁剪操作
            return self.builtins.max(f_min_val, self.builtins.min(float(value), f_max_val))

        # --- 步骤 3: 根据输入类型（列表或标量）执行裁剪 ---
        if isinstance(values, list):
            # [BUGFIX] 直接调用局部函数 `_clip_single`，而不是 `self._clip_single`
            return [_clip_single(v) for v in values]
        else:
            # 如果输入是单个数值，直接调用辅助函数
            return _clip_single(values)
    # 动态导入 math, cmath, builtins 并在实例上提供访问
    # 属性访问器确保这些模块只在需要时被加载
    @property
    def math(self):
        import math
        return math

    @property
    def cmath(self):
        import cmath
        return cmath

    @property
    def builtins(self):
        import builtins
        return builtins

    def _hessenberg_reduction(self, mat: List[List[complex]]) -> Tuple[List[List[complex]], List[List[complex]]]:
        """
        [内部辅助函数] 使用Householder变换将一个厄米矩阵约化为三对角形式。

        Args:
            mat (List[List[complex]]):
                一个 n x n 的厄米矩阵。

        Returns:
            Tuple[List[List[complex]], List[List[complex]]]:
                一个元组 (T, Q)，其中 T 是三对角矩阵，Q 是累积的酉变换矩阵。

        Raises:
            TypeError: 如果输入矩阵格式不正确。
            ValueError: 如果矩阵为空或形状不规整。
        """
        # [健壮性改进] 输入验证
        if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
            self._internal_logger.error(f"_hessenberg_reduction: Input 'mat' must be a non-empty nested list (matrix). Got {type(mat).__name__}.")
            raise TypeError("Input 'mat' must be a non-empty nested list (matrix).")
        n = len(mat)
        if n == 0: return [], []
        if n != len(mat[0]):
            self._internal_logger.error(f"_hessenberg_reduction: Input matrix must be square, but got shape ({n}x{len(mat[0])}).")
            raise ValueError("Input matrix must be square for Hessenberg reduction.")
        
        q = self.eye(n)
        a = [row[:] for row in mat]

        for k in range(n - 2):
            x = [a[i][k] for i in range(k + 1, n)]
            x_norm_sq = self.sum(list(self.abs(val)**2 for val in x)).real # <<< 修正：将生成器转换为列表并取实部

            if self.isclose(x_norm_sq, 0.0):
                continue
            
            x_norm = self.sqrt(x_norm_sq)
            
            phase = x[0] / self.abs(x[0]) if not self.isclose(self.abs(x[0]), 0.0) else 1.0 + 0j
            
            v = [0.0 + 0.0j] * len(x)
            v[0] = x[0] + phase * x_norm
            for i in range(1, len(x)):
                v[i] = x[i]

            v_norm_sq_complex = self.sum(list(self.abs(val)**2 for val in v)) # <<< 修正：将生成器转换为列表
            v_norm_sq = v_norm_sq_complex.real # <<< 修正：取实部
            if self.isclose(v_norm_sq, 0.0):
                continue
            
            v_outer_v_conj = self.outer(v, [val.conjugate() for val in v])
            
            sub_p = self.eye(len(x)) # Dim is len(x) = n - k - 1
            for r in range(len(v)):
                for c in range(len(v)):
                    sub_p[r][c] -= 2 * v_outer_v_conj[r][c] / v_norm_sq
            
            p_full = self.eye(n)
            for r in range(len(x)):
                for c in range(len(x)):
                    p_full[k + 1 + r][k + 1 + c] = sub_p[r][c]

            a = self.dot(self.dot(p_full, a), p_full)
            q = self.dot(q, p_full)
        
        return a, q
    
    
    
    def _qr_decomposition_tridiagonal(self, mat: List[List[complex]]) -> Tuple[List[List[complex]], List[List[complex]]]:
        """
        [内部辅助函数] 使用Givens旋转对一个三对角矩阵进行高效的QR分解。

        Args:
            mat (List[List[complex]]):
                一个 n x n 的厄米三对角矩阵。

        Returns:
            Tuple[List[List[complex]], List[List[complex]]]:
                一个元组 (Q, R)，其中 Q 是酉矩阵，R 是上三角矩阵，
                满足 mat ≈ Q @ R。

        Raises:
            TypeError: 如果输入矩阵格式不正确。
            ValueError: 如果矩阵为空或形状不规整。
        """
        # [健壮性改进] 输入验证
        if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
            self._internal_logger.error(f"_qr_decomposition_tridiagonal: Input 'mat' must be a non-empty nested list (matrix). Got {type(mat).__name__}.")
            raise TypeError("Input 'mat' must be a non-empty nested list (matrix).")
        n = len(mat)
        if n == 0: return [], []
        if n != len(mat[0]):
            self._internal_logger.error(f"_qr_decomposition_tridiagonal: Input matrix must be square, but got shape ({n}x{len(mat[0])}).")
            raise ValueError("Input matrix must be square for QR decomposition.")

        q_t = self.eye(n)
        r = [row[:] for row in mat]

        for j in range(n - 1):
            x, y = r[j][j], r[j+1][j]
            
            norm = self.sqrt(self.abs(x)**2 + self.abs(y)**2)
            
            if self.isclose(norm, 0.0):
                continue
            
            c = x / norm
            s = y / norm
            
            g_dagger = [[c.conjugate(), s.conjugate()], [-s, c]]

            for k in range(j, n):
                r_jk = r[j][k]
                r_jp1k = r[j+1][k]
                
                r[j][k]   = g_dagger[0][0] * r_jk + g_dagger[0][1] * r_jp1k
                r[j+1][k] = g_dagger[1][0] * r_jk + g_dagger[1][1] * r_jp1k

            for k in range(n):
                q_t_jk = q_t[j][k]
                q_t_jp1k = q_t[j+1][k]

                q_t[j][k]   = g_dagger[0][0] * q_t_jk + g_dagger[0][1] * q_t_jp1k
                q_t[j+1][k] = g_dagger[1][0] * q_t_jk + g_dagger[1][1] * q_t_jp1k
        
        q = self.conj_transpose(q_t)
        
        return q, r
    
    def eigh(self, mat: List[List[complex]], max_iterations: int = 1000, tolerance: float = 1e-12) -> Tuple[List[float], List[List[complex]]]:
        """
        [健壮性改进版] 计算一个厄米矩阵的特征值和特征向量。

        Args:
            mat (List[List[complex]]):
                输入的 n x n 厄米矩阵。
            max_iterations (int, optional):
                QR迭代的最大次数，防止无限循环。默认为 1000。
            tolerance (float, optional):
                用于判断次对角线元素是否足够小（收敛）的容差。默认为 1e-12。

        Returns:
            Tuple[List[float], List[List[complex]]]:
                一个元组，第一个元素是包含所有特征值（实数）的列表，
                第二个元素是一个 n x n 矩阵，其列是归一化的特征向量。
        
        Raises:
            ValueError: 如果输入矩阵不是方阵或厄米矩阵。
            TypeError: 如果输入矩阵格式不正确。
        """
        # [健壮性改进] 输入验证
        if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
            self._internal_logger.error(f"eigh: Input 'mat' must be a non-empty nested list (matrix). Got {type(mat).__name__}.")
            raise TypeError("Input 'mat' must be a non-empty nested list (matrix).")
        n = len(mat)
        if n == 0: return [], []
        if n != len(mat[0]):
            self._internal_logger.error(f"eigh: Input matrix must be square, but got shape ({n}x{len(mat[0])}).")
            raise ValueError("Input matrix must be square for eigh.")
        
        if not self.allclose(mat, self.conj_transpose(mat), atol=1e-9):
            self._internal_logger.error("eigh: Input matrix must be Hermitian.")
            raise ValueError("Input matrix must be Hermitian for eigh.")
        
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            self._internal_logger.warning(f"eigh: 'max_iterations' must be a positive integer, got {max_iterations}. Using default 1000.")
            max_iterations = 1000
        if not isinstance(tolerance, (float, int)) or tolerance < 0:
            self._internal_logger.warning(f"eigh: 'tolerance' must be a non-negative number, got {tolerance}. Using default 1e-12.")
            tolerance = 1e-12

        t, q_reduce = self._hessenberg_reduction(mat)
        
        q_total_iter = self.eye(n)
        for i in range(max_iterations):
            q_k, r_k = self._qr_decomposition_tridiagonal(t)
            t = self.dot(r_k, q_k)
            q_total_iter = self.dot(q_total_iter, q_k)
            
            off_diagonal_sum = self.sum(self.abs(t[idx+1][idx]) for idx in range(n-1))
            if off_diagonal_sum < tolerance:
                self._internal_logger.debug(f"eigh: QR iteration converged after {i+1} iterations. Off-diagonal sum: {off_diagonal_sum:.2e}.")
                break
        else:
            self._internal_logger.warning(f"eigh: QR iteration did not converge after max_iterations={max_iterations}. Off-diagonal sum: {off_diagonal_sum:.2e}.")
        
        eigenvalues = [val.real for val in self.diag(t)]
        
        eigenvectors_matrix = self.dot(q_reduce, q_total_iter)
        
        eig_pairs = sorted(zip(eigenvalues, self.transpose(eigenvectors_matrix)))
        
        sorted_eigenvalues = [pair[0] for pair in eig_pairs]
        sorted_eigenvectors_cols = [pair[1] for pair in eig_pairs]
        
        final_eigenvectors_matrix = self.transpose(sorted_eigenvectors_cols)
        
        return sorted_eigenvalues, final_eigenvectors_matrix
    
    def _qr_decomposition_hessenberg(self, mat: List[List[complex]]) -> Tuple[Optional[List[List[complex]]], List[List[complex]]]:
        """
        [内部辅助函数] 使用Givens旋转对一个上海森堡矩阵进行QR分解。
        此方法用于 `eigvalsh`，因为它只需要 R 矩阵，不需要返回完整的 Q 矩阵。

        Args:
            mat (List[List[complex]]):
                一个 n x n 的上海森堡矩阵。

        Returns:
            Tuple[Optional[List[List[complex]]], List[List[complex]]]:
                一个元组 (Q, R)，其中 Q 为 `None`，R 是上三角矩阵。

        Raises:
            TypeError: 如果输入矩阵格式不正确。
            ValueError: 如果矩阵为空或形状不规整。
        """
        # [健壮性改进] 输入验证
        if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
            self._internal_logger.error(f"_qr_decomposition_hessenberg: Input 'mat' must be a non-empty nested list (matrix). Got {type(mat).__name__}.")
            raise TypeError("Input 'mat' must be a non-empty nested list (matrix).")
        n = len(mat)
        if n == 0: return None, []
        if n != len(mat[0]):
            self._internal_logger.error(f"_qr_decomposition_hessenberg: Input matrix must be square, but got shape ({n}x{len(mat[0])}).")
            raise ValueError("Input matrix must be square for QR decomposition.")

        r = [row[:] for row in mat]

        for j in range(n - 1):
            for i in range(j + 1, n):
                if not self.isclose(r[i][j], 0.0):
                    x, y = r[j][j], r[i][j]
                    norm = self.sqrt(self.abs(x)**2 + self.abs(y)**2)
                    
                    if self.isclose(norm, 0.0):
                        continue
                    
                    c = x / norm
                    s = y / norm
                    
                    for k in range(j, n):
                        r_j_k, r_i_k = r[j][k], r[i][k]
                        r[j][k] = c.conjugate() * r_j_k + s.conjugate() * r_i_k
                        r[i][k] = -s * r_j_k + c * r_i_k
        return None, r # 返回 None 作为 Q，因为这里不需要

    def eigvalsh(self, mat: List[List[complex]], max_iterations: int = 1000, tolerance: float = 1e-12) -> List[float]:
        """
        [健壮性改进版] 计算一个厄米矩阵的特征值。

        Args:
            mat (List[List[complex]]):
                输入的厄米矩阵。
            max_iterations (int, optional):
                QR迭代的最大次数，防止无限循环。默认为 1000。
            tolerance (float, optional):
                用于判断次对角线元素是否足够小（收敛）的容差。默认为 1e-12。

        Returns:
            List[float]:
                一个包含矩阵所有特征值（实数）的列表。

        Raises:
            TypeError: 如果输入矩阵格式不正确。
            ValueError: 如果输入矩阵不是方阵或厄米矩阵。
        """
        # [健壮性改进] 输入验证
        if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
            self._internal_logger.error(f"eigvalsh: Input 'mat' must be a non-empty nested list (matrix). Got {type(mat).__name__}.")
            raise TypeError("Input 'mat' must be a non-empty nested list (matrix).")
        rows, cols = len(mat), len(mat[0])
        if rows != cols:
            self._internal_logger.error(f"eigvalsh: Input matrix must be square, but got shape ({rows}x{cols}).")
            raise ValueError(f"Input matrix must be square, but got shape ({rows}x{cols}).")
        if not self.allclose(mat, self.conj_transpose(mat), atol=1e-9):
            self._internal_logger.error("eigvalsh: Input matrix must be Hermitian.")
            raise ValueError("Input matrix must be Hermitian.")

        if not isinstance(max_iterations, int) or max_iterations <= 0:
            self._internal_logger.warning(f"eigvalsh: 'max_iterations' must be a positive integer, got {max_iterations}. Using default 1000.")
            max_iterations = 1000
        if not isinstance(tolerance, (float, int)) or tolerance < 0:
            self._internal_logger.warning(f"eigvalsh: 'tolerance' must be a non-negative number, got {tolerance}. Using default 1e-12.")
            tolerance = 1e-12

        a, _ = self._hessenberg_reduction(mat)
        n = len(a)

        for i in range(max_iterations):
            q_k, r_k = self._qr_decomposition_tridiagonal(a) # 使用三对角 QR 分解
            a = self.dot(r_k, q_k)

            off_diagonal_sum = self.sum(list(self.abs(a[idx+1][idx]) for idx in range(n-1))).real # <<< 修正：将生成器转换为列表并取实部
            if off_diagonal_sum < tolerance:
                self._internal_logger.debug(f"eigvalsh: QR iteration converged after {i+1} iterations. Off-diagonal sum: {off_diagonal_sum:.2e}.")
                break
        else:
            self._internal_logger.warning(f"eigvalsh: QR iteration did not converge after max_iterations={max_iterations}. Off-diagonal sum: {off_diagonal_sum:.2e}.")
        
        eigenvalues = [val.real for val in self.diag(a)]
        
        return sorted(eigenvalues)



class CuPyBackendWrapper:
    """
    [健壮性改进版] [v1.5.13 builtins fix] 一个包装器，用于将CuPy模块的功能适配到与PurePythonBackend相同的接口。
    此版本通过添加 'builtins' 属性，修复了与 PurePythonBackend 的接口不一致问题。

    此类在初始化时会严格检查 `cupy` 模块的可用性。它将所有线性代数和数学操作
    转发给底层的 `cupy` 库，并处理输入/输出的数据类型转换，以确保与抽象接口
    定义的行为一致。
    """
    import random # 用于 choice 函数的 fallback
    from typing import Optional, List, Tuple, Dict, Any, Union

    def __init__(self, cp_module: Any):
        """
        [最终修正增强版] 初始化 CuPyBackendWrapper 实例。

        此构造函数负责验证传入的 `cupy` 模块的可用性和有效性，并将其
        存储为内部引用，以便后续的所有方法调用都能通过它来访问 CuPy 的功能。

        核心增强功能:
        -   **严格的模块验证**:
            -   检查 `cp_module` 是否为 `None`，这是 CuPy 未成功导入的直接标志。
            -   检查 `cp_module` 是否是一个模块（`types.ModuleType`），
                确保传入的是模块对象而不是其他类型。
            -   **功能探测 (Feature Probing)**: 检查 `cp_module` 是否包含了
                一些核心的、后续操作必需的属性或函数（如 `ndarray`, `dot`, `kron`），
                以确保传入的是一个功能完备的 CuPy 模块，而不是一个不完整的
                或伪造的对象。
        -   **清晰的错误处理**: 在验证失败时，会记录详细的 `CRITICAL` 级别
            日志，并抛出明确的 `ImportError` 或 `TypeError`，立即中止
            不正确的初始化过程。
        -   **详细的日志记录**: 初始化成功后，会记录一条 `INFO` 级别的日志，
            确认 CuPy 后端已成功启用，并可以尝试获取并记录 CuPy 的版本号，
            这对于调试环境问题非常有帮助。

        Args:
            cp_module (Any):
                已导入的 `cupy` 模块对象。

        Raises:
            ImportError: 如果 `cp_module` 为 `None`，表示 CuPy 未成功导入。
            TypeError: 如果 `cp_module` 不是一个有效的 CuPy 模块对象。
        """
        # --- 步骤 1: 初始化日志器 ---
        # 确保在进行任何操作之前，日志记录器都已准备就绪。
        self._internal_logger: logging.Logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # --- 步骤 2: 对传入的 cp_module 进行多层次的严格验证 ---

        # 验证 2a: 检查是否为 None
        if cp_module is None:
            self._internal_logger.critical("CuPyBackendWrapper cannot be initialized because the provided 'cp_module' is None. This indicates that the CuPy library failed to import.")
            raise ImportError("CuPy module must be provided to CuPyBackendWrapper. It appears CuPy failed to import.")
        
        # 验证 2b: 检查是否是模块类型
        if not isinstance(cp_module, types.ModuleType):
            self._internal_logger.critical(f"CuPyBackendWrapper expects a module object for 'cp_module', but received an object of type '{type(cp_module).__name__}'.")
            raise TypeError(f"Provided 'cp_module' is not a valid module object. Type received: {type(cp_module).__name__}.")

        # 验证 2c: 功能探测，检查是否存在一些关键属性
        required_attributes = ['ndarray', 'dot', 'kron', 'eye', 'trace', 'linalg', 'random']
        missing_attributes = [attr for attr in required_attributes if not hasattr(cp_module, attr)]
        
        if missing_attributes:
            self._internal_logger.critical(
                f"The provided module for CuPyBackendWrapper is missing essential attributes: {', '.join(missing_attributes)}. "
                "This suggests it is not a valid or complete CuPy installation."
            )
            raise TypeError(f"The provided module is not a valid CuPy module. Missing attributes: {', '.join(missing_attributes)}.")

        # --- 步骤 3: 存储引用并完成初始化 ---
        
        self._cp = cp_module # 存储对实际 CuPy 模块的引用
        
        # 为日志和调试设置一个名称
        self.__name__ = "CuPyBackendWrapper"
        
        # 尝试获取并记录 CuPy 版本号
        cupy_version = getattr(cp_module, '__version__', 'unknown')
        
        self._internal_logger.info(f"CuPyBackendWrapper initialized successfully with CuPy version {cupy_version}. GPU acceleration is enabled.")

    # --- [核心修复 1：添加 math, cmath, 和 builtins 属性以保持接口一致] ---
    @property
    def math(self):
        """
        提供与 `math` 模块兼容的接口，实际调用 `cupy` 的数学函数。
        """
        return self._cp

    @property
    def cmath(self):
        """
        提供与 `cmath` 模块兼容的接口，实际调用 `cupy` 的复数数学函数。
        """
        return self._cp
    
    @property
    def builtins(self):
        """
        提供对标准 `builtins` 模块的访问，以与 PurePythonBackend 保持接口一致。
        """
        import builtins
        return builtins
   
    
    def _ensure_cupy_array(self, arr: Any, dtype: Any = complex) -> Any:
        """
        [内部辅助方法] [v1.5.4 修复] 确保输入是一个指定数据类型的 CuPy 数组。
        此版本修复了当 dtype=None 时的崩溃问题，并增强了类型推断和错误处理。

        此方法是 `CuPyBackendWrapper` 的核心工具之一，负责将来自上层逻辑的、
        可能是多种格式（如 Python 列表、其他类型的 CuPy 数组）的数据，
        统一转换为后续 CuPy 计算所需的 `cupy.ndarray` 对象。

        Args:
            arr (Any):
                输入数据。可以是 Python 标量 (`int`, `float`, `complex`)、
                列表、嵌套列表，或是一个 `cupy.ndarray`。
            dtype (Any, optional):
                期望的 CuPy 数组的数据类型。如果为 `None`，CuPy将自动推断类型。
                默认为 `complex` (即 `cupy.complex128`)。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，其数据内容与输入 `arr` 相同。

        Raises:
            TypeError: 如果输入 `arr` 无法被安全地转换为一个 CuPy 数组。
        """
        # --- [核心修复：处理 dtype=None 的情况] ---
        # 如果 dtype 为 None，意味着我们不应强制转换类型，而是保留或自动推断。
        if dtype is None:
            if isinstance(arr, self._cp.ndarray):
                return arr  # 如果已经是cupy数组，直接返回，不改变其类型。
            try:
                # 如果不是cupy数组，则创建一个新数组，让cupy自动推断最合适的数据类型。
                return self._cp.array(arr)
            except (ValueError, TypeError) as e:
                input_type_name = type(arr).__name__
                self._internal_logger.error(
                    f"Failed to convert input of type '{input_type_name}' to a CuPy array with auto-inferred dtype. "
                    f"This may be due to an irregular shape in a nested list or non-numeric data. CuPy Error: {e}",
                    exc_info=True
                )
                raise TypeError(
                    f"Could not convert input of type '{input_type_name}' to a CuPy array. "
                    f"Please ensure the data is numeric and has a regular shape."
                ) from e
        # --- [修复结束] ---

        # 如果 dtype 不是 None，则执行原始的强制类型转换逻辑
        
        # 步骤 1: 检查输入是否已经是目标类型的 CuPy 数组
        if isinstance(arr, self._cp.ndarray) and arr.dtype == dtype:
            return arr

        # 步骤 2: 检查输入是否是 CuPy 数组但类型不匹配
        if isinstance(arr, self._cp.ndarray):
            try:
                return arr.astype(dtype)
            except Exception as e:
                self._internal_logger.error(
                    f"Failed to change the dtype of an existing CuPy array from {arr.dtype} to {dtype}. Error: {e}",
                    exc_info=True
                )
                raise TypeError(f"Could not convert existing CuPy array of dtype {arr.dtype} to {dtype}.") from e

        # 步骤 3: 处理 Python 数据结构和其他可转换类型
        try:
            return self._cp.array(arr, dtype=dtype)
        except (ValueError, TypeError) as e:
            input_type_name = type(arr).__name__
            self._internal_logger.error(
                f"Failed to convert input of type '{input_type_name}' to a CuPy array with dtype '{dtype}'. "
                f"This may be due to an irregular shape in a nested list or non-numeric data. CuPy Error: {e}",
                exc_info=True
            )
            raise TypeError(
                f"Could not convert input of type '{input_type_name}' to a CuPy array. "
                f"Please ensure the data is numeric and has a regular shape."
            ) from e
        except Exception as e:
            input_type_name = type(arr).__name__
            self._internal_logger.critical(
                f"An unexpected error occurred while converting input of type '{input_type_name}' to a CuPy array. Error: {e}",
                exc_info=True
            )
            raise TypeError(f"An unexpected error occurred during CuPy array conversion.") from e
    
    
    
    
    def _scalar_or_array(self, res: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 将 0 维数组转换为 Python 标量，否则返回原始数组。

        此方法的主要目的是统一 CuPy/NumPy 操作的返回类型。在这些库中，
        对数组进行聚合操作（如 `sum` 一个只包含单个元素的数组）或从数组中
        提取单个元素时，结果通常是一个 0 维数组（例如 `cupy.array(5)`)，
        而不是一个 Python 的原生标量（如 `int` 或 `float`）。

        为了保持后端接口的一致性，并使得上层代码可以安全地对结果进行
        标准的 Python 数值运算，此方法负责将这些 0 维数组“解包”为其
        内部的 Python 标量。

        核心增强功能:
        -   **通用性**: 通过检查 `.ndim` 和 `.item()` 属性，此方法可以同时
            正确处理 CuPy 数组和 NumPy 数组，增强了代码的通用性。
        -   **健壮性**: 在调用 `.item()` 之前，会检查 `res` 是否确实具有
            `ndim` 属性，避免了对非数组类型（如 Python 列表或 `None`）
            调用此方法时产生 `AttributeError`。
        -   **直通行为 (Pass-through)**: 如果输入不是一个 0 维数组（例如，
            它是一个多维数组、一个 Python 标量或 `None`），此方法会直接
            返回原始输入，确保其行为是安全和可预测的。
        -   **清晰的文档**: 详细的文档字符串解释了此方法的动机和行为。

        Args:
            res (Any):
                一个可能是 CuPy/NumPy 数组或任何其他类型的输入值。

        Returns:
            Any:
                如果 `res` 是一个 0 维数组，则返回其内部的 Python 标量。
                否则，返回原始的 `res` 对象。
        """
        # --- 步骤 1: 检查输入是否是一个类数组对象（具有 ndim 属性）---
        # 这是最安全的方式，可以同时处理 CuPy 和 NumPy 数组，而无需显式地
        # 检查 `isinstance(res, (self._cp.ndarray, np.ndarray))`。
        if hasattr(res, 'ndim'):
            
            # --- 步骤 2: 如果是类数组对象，检查其维度是否为 0 ---
            if res.ndim == 0:
                
                # --- 步骤 3: 如果是 0 维数组，调用 .item() 将其转换为 Python 标量 ---
                try:
                    return res.item()
                except Exception as e:
                    # 在极少数情况下，.item() 调用也可能失败。
                    self._internal_logger.warning(
                        f"Failed to convert a 0-dimensional array to a scalar using .item(). Error: {e}. "
                        f"Returning the array object itself.",
                        exc_info=True
                    )
                    return res
            else:
                # 如果是多维数组，则直接返回它
                return res
        else:
            # --- 步骤 4: 如果输入不是类数组对象，则直接返回原始输入 ---
            # 这确保了函数对于 Python 标量、列表、None 等类型的行为是安全的。
            return res

    # --- 基础矩阵/向量创建 ---

    def create_matrix(self, rows: int, cols: int, value: complex = 0.0 + 0.0j) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 创建一个由 CuPy 数组表示的、
        指定大小和初始值的复数矩阵。

        此方法是对 `cupy.full` 函数的一个健壮封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **严格的输入验证**: 在调用底层的 `cupy.full` 之前，对所有输入
            参数 (`rows`, `cols`, `value`) 的类型和值范围进行严格检查。
            -   确保 `rows` 和 `cols` 是非负整数。
            -   确保 `value` 是一个数值类型（`int`, `float`, `complex`）。
        -   **清晰的错误处理**: 当验证失败时，会记录详细的 `ERROR` 级别日志，
            并抛出明确的 `ValueError` 或 `TypeError`，立即中止不正确的
            操作，防止将无效参数传递给 CuPy 库，从而避免可能产生的、
            更晦涩的底层 CUDA 错误。
        -   **显式的数据类型**: 在调用 `cupy.full` 时，明确指定 `dtype=complex`，
            确保生成的数组始终是我们期望的复数类型，增加了代码的确定性。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            rows (int):
                矩阵的行数。必须是非负整数。
            cols (int):
                矩阵的列数。必须是非负整数。
            value (complex, optional):
                矩阵中每个元素的初始值。默认为 `0.0 + 0.0j`。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，代表新创建的复数矩阵。

        Raises:
            ValueError: 如果 `rows` 或 `cols` 为负数。
            TypeError: 如果 `rows`, `cols` 不是整数，或 `value` 不是数值类型。
        """
        # --- 步骤 1: 对输入参数进行严格的验证 ---
        
        # 验证 'rows'
        if not isinstance(rows, int):
            self._internal_logger.error(f"create_matrix: 'rows' must be an integer, but got {type(rows).__name__}.")
            raise TypeError("'rows' must be an integer.")
        if rows < 0:
            self._internal_logger.error(f"create_matrix: 'rows' cannot be negative, but got {rows}.")
            raise ValueError("'rows' cannot be negative.")

        # 验证 'cols'
        if not isinstance(cols, int):
            self._internal_logger.error(f"create_matrix: 'cols' must be an integer, but got {type(cols).__name__}.")
            raise TypeError("'cols' must be an integer.")
        if cols < 0:
            self._internal_logger.error(f"create_matrix: 'cols' cannot be negative, but got {cols}.")
            raise ValueError("'cols' cannot be negative.")

        # 验证 'value'
        if not isinstance(value, (complex, float, int)):
            self._internal_logger.error(f"create_matrix: Initial 'value' must be a numeric type, but got {type(value).__name__}.")
            raise TypeError("Initial 'value' must be a numeric type.")
        
        # --- 步骤 2: 调用底层的 CuPy 函数来创建数组 ---
        try:
            # 使用 cupy.full 函数，它专门用于创建填充了特定值的数组。
            # 明确指定 dtype=complex 以确保类型正确性。
            return self._cp.full((rows, cols), value, dtype=complex)
        
        except Exception as e:
            # --- 步骤 3: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.full while trying to create a matrix of shape ({rows}, {cols}). Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to create CuPy matrix of shape ({rows}, {cols}).") from e

    def eye(self, dim: int, dtype=None) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 创建一个由 CuPy 数组表示的、
        指定维度的复数单位矩阵 (Identity Matrix)。

        此方法是对 `cupy.eye` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。单位矩阵是一个方阵，
        其主对角线上的元素为 1，所有其他元素为 0。

        核心增强功能:
        -   **严格的输入验证**: 在调用底层的 `cupy.eye` 之前，对 `dim` 参数的
            类型和值范围进行严格检查，确保它是一个非负整数。
        -   **清晰的错误处理**: 当验证失败时，会记录详细的 `ERROR` 级别日志，
            并抛出明确的 `ValueError` 或 `TypeError`，立即中止不正确的
            操作，防止将无效参数传递给 CuPy 库。
        -   **显式的数据类型**: 在调用 `cupy.eye` 时，明确指定 `dtype=complex`，
            确保生成的单位矩阵始终是我们期望的复数类型 (`1.0 + 0.0j` on
            the diagonal)，增加了代码的确定性。
        -   **忽略 `dtype` 参数**: 文档明确说明，为了与 NumPy/CuPy API 保持
            兼容性，`dtype` 参数会被接受但被忽略，返回类型始终是 `complex`。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            dim (int):
                方阵的维度（行数或列数）。必须是非负整数。
            dtype (Any, optional):
                此参数为了与 NumPy/CuPy API 保持兼容而被接受，但会被忽略。
                返回的矩阵元素类型始终是 `complex`。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，代表新创建的复数单位矩阵。

        Raises:
            ValueError: 如果 `dim` 为负数。
            TypeError: 如果 `dim` 不是整数。
        """
        # --- 步骤 1: 对输入参数进行严格的验证 ---

        # 验证 'dim' 的类型
        if not isinstance(dim, int):
            self._internal_logger.error(f"eye: Dimension 'dim' must be an integer, but got {type(dim).__name__}.")
            raise TypeError("'dim' must be an integer.")
        
        # 验证 'dim' 的值范围
        if dim < 0:
            self._internal_logger.error(f"eye: Dimension 'dim' cannot be negative, but got {dim}.")
            raise ValueError("'dim' cannot be negative.")

        # --- 步骤 2: 调用底层的 CuPy 函数来创建数组 ---
        try:
            # 使用 cupy.eye 函数创建单位矩阵。
            # 明确指定 dtype=complex 以确保类型正确性，即使 CuPy 默认可能
            # 会根据上下文创建 float64 类型的单位矩阵。
            return self._cp.eye(dim, dtype=complex)
        
        except Exception as e:
            # --- 步骤 3: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.eye while trying to create an identity matrix of dimension {dim}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to create CuPy identity matrix of dimension {dim}.") from e

    def zeros(self, shape: Tuple[int, ...], dtype=None) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 创建一个由 CuPy 数组表示的、
        指定形状的全零向量或矩阵。

        此方法是对 `cupy.zeros` 函数的一个健壮封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **严格的输入验证**: 在调用底层的 `cupy.zeros` 之前，对 `shape` 参数
            的类型和内容进行严格检查。
            -   确保 `shape` 是一个元组。
            -   确保 `shape` 元组中的所有元素都是非负整数。
        -   **清晰的错误处理**: 当验证失败时，会记录详细的 `ERROR` 级别日志，
            并抛出明确的 `ValueError` 或 `TypeError`，立即中止不正确的
            操作，防止将无效参数传递给 CuPy 库。
        -   **显式的数据类型**: 在调用 `cupy.zeros` 时，明确指定 `dtype=complex`，
            确保生成的数组始终是我们期望的复数类型 (`0.0 + 0.0j`)，增加了
            代码的确定性。
        -   **忽略 `dtype` 参数**: 文档明确说明，为了与 NumPy/CuPy API 保持
            兼容性，`dtype` 参数会被接受但被忽略，返回类型始终是 `complex`。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            shape (Tuple[int, ...]):
                一个元组，指定了输出数组的维度，例如 `(n,)` 或 `(m, n)`。
                元组中的所有元素都必须是非负整数。
            dtype (Any, optional):
                此参数为了与 NumPy/CuPy API 保持兼容而被接受，但会被忽略。
                返回的数组元素类型始终是 `complex`。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，代表新创建的、所有元素均为
                `0.0 + 0.0j` 的复数数组。

        Raises:
            ValueError: 如果 `shape` 元组中包含负数。
            TypeError: 如果 `shape` 不是一个元组，或其元素不是整数。
        """
        # --- 步骤 1: 对输入参数进行严格的验证 ---

        # 验证 'shape' 的类型
        if not isinstance(shape, tuple):
            self._internal_logger.error(f"zeros: 'shape' must be a tuple, but got {type(shape).__name__}.")
            raise TypeError("'shape' must be a tuple.")
        
        # 验证 'shape' 的内容
        for dimension in shape:
            if not isinstance(dimension, int):
                self._internal_logger.error(f"zeros: All elements in 'shape' tuple must be integers, but found '{dimension}' of type {type(dimension).__name__} in {shape}.")
                raise TypeError(f"All elements in 'shape' tuple must be integers. Found {type(dimension).__name__}.")
            if dimension < 0:
                self._internal_logger.error(f"zeros: Dimensions in 'shape' tuple cannot be negative, but found '{dimension}' in {shape}.")
                raise ValueError(f"Dimensions in 'shape' tuple cannot be negative. Found {dimension}.")

        # --- 步骤 2: 调用底层的 CuPy 函数来创建数组 ---
        try:
            # 使用 cupy.zeros 函数创建全零数组。
            # 明确指定 dtype=complex 以确保类型正确性。
            return self._cp.zeros(shape, dtype=complex)
        
        except Exception as e:
            # --- 步骤 3: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.zeros while trying to create an array of shape {shape}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to create CuPy zeros array of shape {shape}.") from e
        
    def zeros_like(self, arr: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 创建一个与给定数组 `arr` 形状相同
        且类型为 `complex` 的全零 CuPy 数组。

        此方法是对 `cupy.zeros_like` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.zeros_like` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `arr` 安全地转换为一个 CuPy 数组。
            这使得此方法不仅能处理 CuPy 数组，还能无缝地处理 Python 列表、
            嵌套列表等多种输入格式。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败（例如，因为
            输入的 Python 列表形状不规则），`zeros_like` 会捕获该 `TypeError`
            并提供一个带有上下文的、更明确的错误信息。
        -   **显式的数据类型**: 在调用 `cupy.zeros_like` 时，明确指定 `dtype=complex`，
            确保生成的数组始终是我们期望的复数类型 (`0.0 + 0.0j`)，增加了
            代码的确定性。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            arr (Any):
                一个类数组对象，其形状将被用于创建新的全零数组。可以是
                `cupy.ndarray`、Python 列表、嵌套列表等。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，其形状与 `arr` 相同，所有元素均为
                `0.0 + 0.0j`。

        Raises:
            TypeError: 如果输入 `arr` 无法被转换为一个有效的 CuPy 数组
                       （例如，形状不规则的嵌套列表）。
        """
        # --- 步骤 1: 将输入安全地转换为一个 CuPy 数组 ---
        try:
            # 使用 _ensure_cupy_array 来处理多种输入类型。
            # 我们不需要关心转换后的 dtype，因为 cupy.zeros_like 只关心形状。
            # 为了效率，如果 arr 已经是 CuPy 数组，这里会很快返回。
            arr_cupy = self._ensure_cupy_array(arr)
        except TypeError as e:
            # 如果转换失败，说明输入 `arr` 的格式有问题（例如，不规则的列表）。
            self._internal_logger.error(
                f"zeros_like: The input 'arr' of type {type(arr).__name__} could not be converted to a CuPy array, "
                "so its shape could not be determined. It might be an irregular nested list.",
                exc_info=True
            )
            raise TypeError(
                f"Input 'arr' for zeros_like cannot be processed. Please provide a CuPy array or a regular nested list."
            ) from e

        # --- 步骤 2: 调用底层的 CuPy 函数来创建数组 ---
        try:
            # 使用 cupy.zeros_like 函数，它会自动匹配 arr_cupy 的形状。
            # 明确指定 dtype=complex 以确保类型正确性。
            return self._cp.zeros_like(arr_cupy, dtype=complex)
        
        except Exception as e:
            # --- 步骤 3: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            shape_str = str(arr_cupy.shape) if hasattr(arr_cupy, 'shape') else "unknown"
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.zeros_like for an array of shape {shape_str}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to create CuPy zeros_like array for shape {shape_str}.") from e

    # --- 核心线性代数操作 ---

    def dot(self, mat_a: Any, mat_b: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算两个类数组对象的矩阵乘法 (dot product)。

        此方法是对 `cupy.dot` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。它能够处理矩阵-矩阵、
        矩阵-向量以及向量-向量（内积）的乘法。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.dot` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `mat_a` 和 `mat_b` 安全地
            转换为 CuPy 数组。这使得此方法能无缝处理 Python 列表、
            嵌套列表等多种输入格式。
        -   **维度兼容性预检查**: 在调用 `cupy.dot` 之前，会手动检查两个
            数组的内积维度是否匹配。如果不匹配，会记录详细的错误日志并
            抛出一个带有明确形状信息的 `ValueError`。这比直接让 CuPy 抛出
            一个可能更通用的 `ValueError` 提供了更好的调试体验。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败，或者
            `cupy.dot` 自身抛出异常，都会被捕获、记录，并重新包装成带有
            清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            mat_a (Any):
                乘法中的左矩阵或向量。可以是 `cupy.ndarray`、Python 列表、
                嵌套列表等。
            mat_b (Any):
                乘法中的右矩阵或向量。可以是 `cupy.ndarray`、Python 列表、
                嵌套列表等。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，代表矩阵乘法的结果。

        Raises:
            TypeError: 如果输入 `mat_a` 或 `mat_b` 无法被转换为一个有效的 CuPy 数组。
            ValueError: 如果两个数组的内积维度不匹配。
        """
        # --- 步骤 1: 将输入安全地转换为 CuPy 数组 ---
        try:
            # 使用 _ensure_cupy_array 来处理多种输入类型
            a_cp = self._ensure_cupy_array(mat_a)
            b_cp = self._ensure_cupy_array(mat_b)
        except TypeError as e:
            # 如果转换失败，说明输入格式有问题
            self._internal_logger.error(
                f"dot: One or both inputs could not be converted to a CuPy array. "
                f"Input A type: {type(mat_a).__name__}, Input B type: {type(mat_b).__name__}.",
                exc_info=True
            )
            raise TypeError(
                f"Inputs to dot product must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 2: 维度兼容性预检查 ---
        # cupy.dot 要求 a.shape[-1] == b.shape[-2] (如果是多维) 或 a.shape[-1] == b.shape[0] (如果是 2D vs 1D)
        # 我们可以简化这个检查，因为我们主要处理 1D 和 2D 的情况。
        try:
            a_inner_dim = a_cp.shape[-1]
            # b 的内积维度取决于它是 1D 还是 >=2D
            b_inner_dim = b_cp.shape[-2] if b_cp.ndim >= 2 else b_cp.shape[0]

            if a_inner_dim != b_inner_dim:
                self._internal_logger.error(
                    f"dot: Matrix dimensions are not compatible for dot product. "
                    f"Shape of A: {a_cp.shape}, Shape of B: {b_cp.shape}. "
                    f"Inner dimensions ({a_inner_dim} and {b_inner_dim}) must match."
                )
                raise ValueError(
                    f"Matrix dimensions are not compatible for dot product: "
                    f"Shapes {a_cp.shape} and {b_cp.shape} cannot be multiplied."
                )
        except IndexError:
            # 如果 .shape[-1] 或 .shape[-2] 访问失败，说明数组是 0 维或有其他问题
            self._internal_logger.error(
                f"dot: Could not determine inner dimensions for dot product from shapes {a_cp.shape} and {b_cp.shape}. "
                "Inputs might be 0-dimensional arrays or have an unsupported shape.",
                exc_info=True
            )
            raise ValueError(f"Could not perform dot product on inputs with shapes {a_cp.shape} and {b_cp.shape}.")

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 现在可以安全地调用 cupy.dot
            return self._cp.dot(a_cp, b_cp)
        
        except Exception as e:
            # --- 步骤 4: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.dot with input shapes {a_cp.shape} and {b_cp.shape}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to compute dot product for shapes {a_cp.shape} and {b_cp.shape}.") from e

    def kron(self, mat_a: Any, mat_b: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算两个类数组对象的张量积 (Kronecker product)。

        此方法是对 `cupy.kron` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.kron` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `mat_a` 和 `mat_b` 安全地
            转换为 CuPy 数组。这使得此方法能无缝处理 Python 列表、
            嵌套列表等多种输入格式。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败（例如，因为
            输入的 Python 列表形状不规则），`kron` 会捕获该 `TypeError`
            并提供一个带有上下文的、更明确的错误信息。
        -   **健壮的底层调用**: 对 `cupy.kron` 的调用被包裹在 `try...except`
            块中，以捕获任何来自 CuPy 库的意外异常（如内存不足），并将其
            重新包装成带有清晰上下文的 `RuntimeError` 向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            mat_a (Any):
                张量积中的左矩阵或向量。可以是 `cupy.ndarray`、Python 列表、
                嵌套列表等。
            mat_b (Any):
                张量积中的右矩阵或向量。可以是 `cupy.ndarray`、Python 列表、
                嵌套列表等。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，代表张量积的结果。

        Raises:
            TypeError: 如果输入 `mat_a` 或 `mat_b` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 CuPy 数组 ---
        try:
            # 使用 _ensure_cupy_array 来处理多种输入类型。
            # 默认 dtype=complex 确保了即使输入是整数或浮点数，
            # 它们也会被转换为复数数组，这对于量子计算是必需的。
            a_cp = self._ensure_cupy_array(mat_a, dtype=complex)
            b_cp = self._ensure_cupy_array(mat_b, dtype=complex)
        except TypeError as e:
            # 如果转换失败，说明输入格式有问题
            self._internal_logger.error(
                f"kron: One or both inputs could not be converted to a CuPy array. "
                f"Input A type: {type(mat_a).__name__}, Input B type: {type(mat_b).__name__}.",
                exc_info=True
            )
            raise TypeError(
                f"Inputs to Kronecker product must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 2: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 现在可以安全地调用 cupy.kron
            return self._cp.kron(a_cp, b_cp)
        
        except Exception as e:
            # --- 步骤 3: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            # 这可能包括内存不足错误 (cupy.cuda.memory.OutOfMemoryError) 或其他 CUDA 错误。
            shape_a_str = str(a_cp.shape) if hasattr(a_cp, 'shape') else "unknown"
            shape_b_str = str(b_cp.shape) if hasattr(b_cp, 'shape') else "unknown"
            
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.kron with input shapes {shape_a_str} and {shape_b_str}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to compute Kronecker product for shapes {shape_a_str} and {shape_b_str}.") from e

    def outer(self, vec_a: Any, vec_b: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算两个类向量对象的向量外积 (outer product)。

        此方法是对 `cupy.outer` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。外积的结果是一个矩阵 M，
        其元素由 `M[i,j] = a[i] * b[j]` 定义。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.outer` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `vec_a` 和 `vec_b` 安全地
            转换为 CuPy 数组。这使得此方法能无缝处理 Python 列表等多种输入格式。
        -   **维度检查**: 在转换后，会检查确保两个输入数组都是一维的（向量），
            因为 `cupy.outer` 期望的是向量输入。如果不是，会抛出一个明确的
            `ValueError`。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败，或者
            `cupy.outer` 自身抛出异常，都会被捕获、记录，并重新包装成带有
            清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            vec_a (Any):
                外积中的左向量。可以是 `cupy.ndarray`、Python 列表等。
                将被展平为一维向量处理。
            vec_b (Any):
                外积中的右向量。可以是 `cupy.ndarray`、Python 列表等。
                将被展平为一维向量处理。

        Returns:
            Any:
                一个二维的 `cupy.ndarray` 对象，代表外积的结果矩阵。

        Raises:
            TypeError: 如果输入 `vec_a` 或 `vec_b` 无法被转换为一个有效的 CuPy 数组。
            ValueError: 如果转换后的输入数组不是一维的。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 CuPy 数组 ---
        try:
            # 使用 _ensure_cupy_array 来处理多种输入类型。
            # 默认 dtype=complex 确保了即使输入是整数或浮点数，
            # 它们也会被转换为复数数组。
            a_cp = self._ensure_cupy_array(vec_a, dtype=complex)
            b_cp = self._ensure_cupy_array(vec_b, dtype=complex)
        except TypeError as e:
            # 如果转换失败，说明输入格式有问题
            self._internal_logger.error(
                f"outer: One or both inputs could not be converted to a CuPy array. "
                f"Input A type: {type(vec_a).__name__}, Input B type: {type(vec_b).__name__}.",
                exc_info=True
            )
            raise TypeError(
                f"Inputs to outer product must be convertible to a CuPy array (e.g., a regular list)."
            ) from e

        # --- 步骤 2: 确保输入是一维向量 ---
        # cupy.outer 会自动将多维输入展平 (flatten)，但为了接口的明确性和
        # 防止意外行为，我们最好在这里显式地进行检查或展平。
        # 我们将显式展平以匹配 CuPy 的行为。
        if a_cp.ndim > 1:
            self._internal_logger.debug(f"Input 'vec_a' for outer product has {a_cp.ndim} dimensions. Flattening it to 1D.")
            a_cp = a_cp.ravel()
        
        if b_cp.ndim > 1:
            self._internal_logger.debug(f"Input 'vec_b' for outer product has {b_cp.ndim} dimensions. Flattening it to 1D.")
            b_cp = b_cp.ravel()

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 现在可以安全地调用 cupy.outer
            return self._cp.outer(a_cp, b_cp)
        
        except Exception as e:
            # --- 步骤 4: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            len_a_str = str(a_cp.shape[0]) if hasattr(a_cp, 'shape') and a_cp.ndim > 0 else "unknown"
            len_b_str = str(b_cp.shape[0]) if hasattr(b_cp, 'shape') and b_cp.ndim > 0 else "unknown"
            
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.outer with input lengths {len_a_str} and {len_b_str}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to compute outer product for vectors of lengths {len_a_str} and {len_b_str}.") from e

    def conj_transpose(self, mat: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个类数组对象的共轭转置 (Dagger, †)。

        此方法是对 CuPy 数组操作 `.conj().T` 的一个健 robust封装，旨在提供一个
        与 `PurePythonBackend` 中对应方法一致的接口。共轭转置是量子力学中
        一个基础且频繁使用的操作。

        核心增强功能:
        -   **灵活的输入处理**: 在执行操作之前，会先通过 `self._ensure_cupy_array`
            方法将输入 `mat` 安全地转换为 CuPy 数组。这使得此方法能无缝处理
            Python 列表、嵌套列表等多种输入格式。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败（例如，因为
            输入的 Python 列表形状不规则），`conj_transpose` 会捕获该 `TypeError`
            并提供一个带有上下文的、更明确的错误信息。
        -   **健壮的底层调用**: 对 `.conj().T` 的调用被包裹在 `try...except`
            块中，以捕获任何来自 CuPy 库的意外异常，并将其重新包装成带有
        -   清晰上下文的 `RuntimeError` 向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            mat (Any):
                要进行共轭转置的矩阵或向量。可以是 `cupy.ndarray`、
                Python 列表、嵌套列表等。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，代表输入 `mat` 的共轭转置。

        Raises:
            TypeError: 如果输入 `mat` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 CuPy 数组 ---
        try:
            # 使用 _ensure_cupy_array 来处理多种输入类型。
            # 默认 dtype=complex 确保了即使输入是整数或浮点数，
            # 它们也会被转换为复数数组。
            mat_cp = self._ensure_cupy_array(mat, dtype=complex)
        except TypeError as e:
            # 如果转换失败，说明输入格式有问题
            self._internal_logger.error(
                f"conj_transpose: The input of type {type(mat).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for conj_transpose must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 2: 执行共轭转置操作 ---
        try:
            # .conj() 计算逐元素的复共轭
            # .T 属性获取数组的转置
            # 这是一个链式操作，高效且符合 CuPy 的惯例。
            return mat_cp.conj().T
        
        except Exception as e:
            # --- 步骤 3: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            shape_str = str(mat_cp.shape) if hasattr(mat_cp, 'shape') else "unknown"
            
            self._internal_logger.critical(
                f"An unexpected error occurred in CuPy during conj_transpose for an array of shape {shape_str}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to compute conjugate transpose for an array of shape {shape_str}.") from e

    def trace(self, mat: Any) -> complex:
        """
        [内部辅助方法] [最终修正增强版] 计算一个类数组方阵的迹 (trace)。

        此方法是对 `cupy.trace` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。矩阵的迹是其主对角线上
        所有元素的总和。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.trace` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `mat` 安全地转换为 CuPy 数组。
            这使得此方法能无缝处理 Python 列表、嵌套列表等多种输入格式。
        -   **形状预检查**: 在计算迹之前，会验证转换后的数组是否为二维方阵
            （即 `ndim == 2` 且 `rows == cols`）。如果不是，会记录详细的
            错误日志并抛出一个明确的 `ValueError`。这比直接让 `cupy.trace`
            在非方阵上产生可能不符合预期的结果要安全得多。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败，或者
            `cupy.trace` 自身抛出异常，都会被捕获、记录，并重新包装成带有
            清晰上下文的异常向上层抛出。
        -   **返回类型保证**: 确保返回值总是一个 Python 的 `complex` 标量，
            通过调用 `_scalar_or_array` 和 `complex()` 来实现。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            mat (Any):
                要计算迹的方阵。可以是 `cupy.ndarray`、Python 嵌套列表等。

        Returns:
            complex:
                矩阵的迹，作为一个 Python 的 `complex` 标量。

        Raises:
            TypeError: 如果输入 `mat` 无法被转换为一个有效的 CuPy 数组。
            ValueError: 如果转换后的输入数组不是一个二维方阵。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 CuPy 数组 ---
        try:
            mat_cp = self._ensure_cupy_array(mat)
        except TypeError as e:
            self._internal_logger.error(
                f"trace: The input of type {type(mat).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for trace must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 2: 验证数组是否为二维方阵 ---
        if not hasattr(mat_cp, 'ndim') or mat_cp.ndim != 2:
            shape_str = str(mat_cp.shape) if hasattr(mat_cp, 'shape') else "unknown"
            self._internal_logger.error(f"trace: Input must be a 2D matrix, but got an array with {mat_cp.ndim} dimensions and shape {shape_str}.")
            raise ValueError(f"Trace is only defined for 2D matrices in this context, but got array of shape {shape_str}.")
        
        rows, cols = mat_cp.shape
        if rows != cols:
            self._internal_logger.error(f"trace: Trace is only defined for square matrices, but got a matrix of shape ({rows}, {cols}).")
            raise ValueError(f"Trace is only defined for square matrices, but got shape ({rows}, {cols}).")
        
        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.trace 计算迹
            trace_result = self._cp.trace(mat_cp)
            
            # --- 步骤 4: 确保返回类型是 Python 标量 ---
            # cupy.trace 返回一个 0 维数组，需要转换为 Python 标量
            scalar_trace = self._scalar_or_array(trace_result)

            # 确保最终返回的是 complex 类型
            return complex(scalar_trace)

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            shape_str = str(mat_cp.shape)
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.trace for an array of shape {shape_str}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to compute trace for an array of shape {shape_str}.") from e

    def diag(self, mat: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 提取一个类数组方阵的主对角线元素。

        此方法是对 `cupy.diag` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。在量子计算中，它主要用于
        从密度矩阵中提取测量概率。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.diag` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `mat` 安全地转换为 CuPy 数组。
            这使得此方法能无缝处理 Python 列表、嵌套列表等多种输入格式。
        -   **形状预检查**: 在提取对角线之前，会验证转换后的数组是否为二维方阵
            （即 `ndim == 2` 且 `rows == cols`）。如果不是，会记录详细的
            错误日志并抛出一个明确的 `ValueError`。这确保了函数的行为符合
            在量子计算上下文中的预期。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败，或者
            `cupy.diag` 自身抛出异常，都会被捕获、记录，并重新包装成带有
            清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            mat (Any):
                要提取对角线的方阵。可以是 `cupy.ndarray`、Python 嵌套列表等。

        Returns:
            Any:
                一个一维的 `cupy.ndarray` 对象，包含了输入矩阵 `mat` 的主对角线元素。

        Raises:
            TypeError: 如果输入 `mat` 无法被转换为一个有效的 CuPy 数组。
            ValueError: 如果转换后的输入数组不是一个二维方阵。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 CuPy 数组 ---
        try:
            mat_cp = self._ensure_cupy_array(mat)
        except TypeError as e:
            self._internal_logger.error(
                f"diag: The input of type {type(mat).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for diag must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 2: 验证数组是否为二维方阵 ---
        if not hasattr(mat_cp, 'ndim') or mat_cp.ndim != 2:
            shape_str = str(mat_cp.shape) if hasattr(mat_cp, 'shape') else "unknown"
            self._internal_logger.error(f"diag: Input must be a 2D matrix, but got an array with {mat_cp.ndim} dimensions and shape {shape_str}.")
            raise ValueError(f"Diagonal can only be extracted from a 2D matrix in this context, but got array of shape {shape_str}.")
        
        rows, cols = mat_cp.shape
        if rows != cols:
            self._internal_logger.error(f"diag: Diagonal can only be extracted from a square matrix in this context, but got a matrix of shape ({rows}, {cols}).")
            raise ValueError(f"Diagonal can only be extracted from a square matrix, but got shape ({rows}, {cols}).")
        
        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.diag 提取主对角线
            return self._cp.diag(mat_cp)

        except Exception as e:
            # --- 步骤 4: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            shape_str = str(mat_cp.shape)
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.diag for an array of shape {shape_str}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to extract diagonal for an array of shape {shape_str}.") from e

    def reshape(self, data: Any, new_shape: Tuple[int, ...]) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 将一个类数组对象重塑为新的指定形状。

        此方法是对 `cupy.reshape` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.reshape` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `data` 安全地转换为 CuPy 数组。
            这使得此方法能无缝处理 Python 列表、嵌套列表等多种输入格式。
        -   **严格的形状参数验证**: 在调用 `cupy.reshape` 之前，对 `new_shape`
            参数的类型和内容进行严格检查。
            -   确保 `new_shape` 是一个元组。
            -   确保 `new_shape` 元组中的所有元素都是整数。
            -   确保 `new_shape` 中最多只包含一个 `-1`（用于维度推断）。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败，或者
            `new_shape` 参数无效，或者 `cupy.reshape` 自身因尺寸不兼容而
            抛出异常，都会被捕获、记录，并重新包装成带有清晰上下文的异常
            向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            data (Any):
                要重塑的数据。可以是 `cupy.ndarray`、Python 列表、嵌套列表等。
            new_shape (Tuple[int, ...]):
                一个整数元组，指定了输出数组的新维度。可以包含一个 `-1`
                来自动推断该维度的大小。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，代表重塑后的新数组。

        Raises:
            TypeError: 如果输入 `data` 无法被转换为一个有效的 CuPy 数组，
                       或者 `new_shape` 不是一个元组或其元素不是整数。
            ValueError: 如果 `new_shape` 包含多个 `-1`，或者数据大小与
                        目标形状不兼容。
        """
        # --- 步骤 1: 将输入数据安全地转换为 CuPy 数组 ---
        try:
            data_cp = self._ensure_cupy_array(data)
        except TypeError as e:
            self._internal_logger.error(
                f"reshape: The input 'data' of type {type(data).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input 'data' for reshape must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 2: 对 new_shape 参数进行严格的验证 ---
        if not isinstance(new_shape, tuple):
            self._internal_logger.error(f"reshape: 'new_shape' must be a tuple of integers, but got {type(new_shape).__name__}.")
            raise TypeError("'new_shape' must be a tuple of integers.")

        neg_one_count = 0
        for dimension in new_shape:
            if not isinstance(dimension, int):
                self._internal_logger.error(f"reshape: All elements in 'new_shape' tuple must be integers, but found '{dimension}' of type {type(dimension).__name__} in {new_shape}.")
                raise TypeError(f"All elements in 'new_shape' tuple must be integers. Found {type(dimension).__name__}.")
            if dimension == -1:
                neg_one_count += 1
        
        if neg_one_count > 1:
            self._internal_logger.error(f"reshape: 'new_shape' {new_shape} can only contain at most one '-1', but found {neg_one_count}.")
            raise ValueError(f"Cannot reshape: 'new_shape' {new_shape} can only contain at most one '-1'.")

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 现在可以安全地调用 cupy.reshape
            return self._cp.reshape(data_cp, new_shape)
        
        except ValueError as e:
            # CuPy 的 reshape 在尺寸不匹配时会抛出 ValueError。我们捕获它并添加更多上下文。
            self._internal_logger.error(
                f"reshape: Cannot reshape array of size {data_cp.size} into shape {new_shape}. "
                f"The total number of elements must remain the same. CuPy Error: {e}",
                exc_info=True
            )
            raise ValueError(
                f"Cannot reshape array of size {data_cp.size} into shape {new_shape}. "
                "The total number of elements must be constant."
            ) from e
        
        except Exception as e:
            # --- 步骤 4: 捕获并重新包装来自 CuPy 的任何其他意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.reshape with input shape {data_cp.shape} and target shape {new_shape}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to reshape array from {data_cp.shape} to {new_shape}.") from e

    def einsum(self, subscripts: str, *tensors: Any, optimize: bool = True) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 对 CuPy 数组执行爱因斯坦求和约定计算。

        此方法是对 `cupy.einsum` 的一个健 robust封装，主要用于量子态的部分迹
        操作，但也能处理通用的 `einsum` 任务。

        核心增强功能:
        -   **严格的输入验证**: 在执行任何计算之前，对 `subscripts` 字符串的
            格式和 `tensors` 的存在性进行严格检查。
        -   **灵活的输入处理**: 在调用任何底层计算之前，会通过
            `self._ensure_cupy_array` 方法将所有输入的 `tensors` 安全地
            转换为 CuPy 数组。
        -   **双路径执行**:
            1.  **优化路径**: 智能识别用于“从态矢量计算部分迹”的特殊下标格式
                （例如 `"aB,Ab->aA"`）。如果匹配，则调用一个专门为此任务编写的、
                行为与 `PurePythonBackend` 完全一致的硬编码内核
                `_partial_trace_statevector_kernel`。
            2.  **通用路径**: 对于所有其他 `einsum` 任务，安全地回退到调用
                `cupy.einsum`。
        -   **清晰的错误处理**: 无论是特殊内核还是通用路径，所有的计算都被包裹在
            `try...except` 块中。任何来自 CuPy 的异常都会被捕获、记录，并
            重新包装成带有清晰上下文的 `RuntimeError` 向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的双路径行为、参数、
            返回值和可能引发的异常。

        Args:
            subscripts (str):
                一个描述 `einsum` 操作的字符串，例如 `"abAB->aA"` 或 `"aB,Ab->aA"`。
            *tensors (Any):
                一个或多个类数组输入张量（`cupy.ndarray`, Python 列表等）。
            optimize (bool, optional):
                传递给通用 `cupy.einsum` 调用的优化标志。默认为 `True`。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，代表 `einsum` 计算的结果。

        Raises:
            TypeError: 如果 `subscripts` 不是字符串，或输入张量无法转换为 CuPy 数组。
            ValueError: 如果 `subscripts` 格式不正确或没有提供张量。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 对输入参数进行严格的验证 ---
        if not isinstance(subscripts, str) or '->' not in subscripts:
            self._internal_logger.error(f"einsum: Invalid 'subscripts' format: '{subscripts}'. It must be a string containing '->'.")
            raise ValueError("Invalid 'subscripts' format. It must be a string containing '->'.")
        if not tensors:
            self._internal_logger.error("einsum: No input tensors provided.")
            raise ValueError("einsum() requires at least one tensor array.")

        # --- 步骤 2: 识别是否为我们的“从态矢量计算部分迹”的特殊任务 ---
        # 这种任务的特征是：2个输入张量，且下标格式类似于 "aB,Ab->aA"
        is_statevector_partial_trace = False
        if len(tensors) == 2:
            input_subs_part = subscripts.split('->')[0]
            if ',' in input_subs_part:
                # 这是一个启发式检查，可能不够完美，但能覆盖我们的用例。
                # 更严格的检查可以在 `_partial_trace_statevector_kernel` 内部完成。
                is_statevector_partial_trace = True

        # --- 步骤 3: 根据任务类型选择执行路径 ---
        if is_statevector_partial_trace:
            # --- 路径 A: 调用专门优化的部分迹内核 ---
            self._internal_logger.debug(f"einsum: Detected statevector partial trace task. Routing to specialized kernel for subscripts '{subscripts}'.")
            try:
                vec_tensor = self._ensure_cupy_array(tensors[0])
                vec_tensor_conj = self._ensure_cupy_array(tensors[1])
                
                return self._partial_trace_statevector_kernel(vec_tensor, vec_tensor_conj, subscripts)
            
            except Exception as e:
                self._internal_logger.critical(
                    f"An unexpected error occurred in the specialized _partial_trace_statevector_kernel for subscripts '{subscripts}'. Error: {e}",
                    exc_info=True
                )
                raise RuntimeError(f"Failed during specialized partial trace calculation for '{subscripts}'.") from e
        else:
            # --- 路径 B: 调用通用的 cupy.einsum (回退路径) ---
            self._internal_logger.debug(f"einsum: Routing to generic cupy.einsum for subscripts '{subscripts}'.")
            try:
                # 在调用前，确保所有张量都已转换为 CuPy 数组
                processed_tensors = [self._ensure_cupy_array(t) for t in tensors]
                
                return self._cp.einsum(subscripts, *processed_tensors, optimize=optimize)
            
            except TypeError as e:
                # 捕获因张量转换失败而产生的 TypeError
                self._internal_logger.error(
                    f"einsum: One or more input tensors could not be converted to a CuPy array for generic einsum. Error: {e}",
                    exc_info=True
                )
                raise TypeError("Inputs to einsum must be convertible to a CuPy array.") from e
            except Exception as e:
                # 捕获来自 cupy.einsum 的任何其他错误
                tensor_shapes = [t.shape if hasattr(t, 'shape') else 'unknown' for t in tensors]
                self._internal_logger.critical(
                    f"An unexpected error occurred in generic cupy.einsum with subscripts '{subscripts}' and tensor shapes {tensor_shapes}. Error: {e}",
                    exc_info=True
                )
                raise RuntimeError(f"Generic cupy.einsum calculation failed for '{subscripts}'.") from e

    def transpose(self, mat: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个类数组对象的转置。

        此方法是对 CuPy 数组 `.T` 属性访问的一个健 robust封装，旨在提供一个
        与 `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在执行操作之前，会先通过 `self._ensure_cupy_array`
            方法将输入 `mat` 安全地转换为 CuPy 数组。这使得此方法能无缝处理
            Python 列表、嵌套列表等多种输入格式。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败（例如，因为
            输入的 Python 列表形状不规则），`transpose` 会捕获该 `TypeError`
            并提供一个带有上下文的、更明确的错误信息。
        -   **健壮的底层调用**: 对 `.T` 属性的访问被包裹在 `try...except`
            块中，以捕获任何来自 CuPy 库的意外异常，并将其重新包装成带有
            清晰上下文的 `RuntimeError` 向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            mat (Any):
                要进行转置的矩阵或向量。可以是 `cupy.ndarray`、
                Python 列表、嵌套列表等。

        Returns:
            Any:
                一个 `cupy.ndarray` 对象，代表输入 `mat` 的转置。

        Raises:
            TypeError: 如果输入 `mat` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部操作时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 CuPy 数组 ---
        try:
            # 使用 _ensure_cupy_array 来处理多种输入类型。
            mat_cp = self._ensure_cupy_array(mat)
        except TypeError as e:
            # 如果转换失败，说明输入格式有问题
            self._internal_logger.error(
                f"transpose: The input of type {type(mat).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for transpose must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 2: 执行转置操作 ---
        try:
            # .T 属性是获取数组转置的最高效、最符合 CuPy/NumPy 惯例的方式。
            # 对于一维数组 (n,)，.T 返回的仍是 (n,)，行为正确。
            # 对于二维数组 (m,n)，.T 返回的是 (n,m)。
            return mat_cp.T
        
        except Exception as e:
            # --- 步骤 3: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            shape_str = str(mat_cp.shape) if hasattr(mat_cp, 'shape') else "unknown"
            
            self._internal_logger.critical(
                f"An unexpected error occurred in CuPy during transpose for an array of shape {shape_str}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to compute transpose for an array of shape {shape_str}.") from e

    def isclose(self, a: Union[complex, float, int, Any], b: Union[complex, float, int, Any], atol: float = 1e-9) -> bool:
        """
        [内部辅助方法] [最终修正增强版] 判断两个数值（包括 CuPy 0维数组）
        是否在指定的绝对容差内“足够接近”。

        此方法是对 `cupy.isclose` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.isclose` 之前，会通过
            `self._ensure_cupy_array` 方法将输入 `a` 和 `b` 安全地转换为
            CuPy 数组。这使得此方法能无缝处理 Python 标量 (`int`, `float`,
            `complex`)。
        -   **严格的容差验证**: 对 `atol` 参数的类型和值范围进行严格检查，
            确保它是一个非负的浮点数。
        -   **返回类型保证**: 确保返回值总是一个 Python 的 `bool` 标量，
            通过调用 `_scalar_or_array` 和 `bool()` 来实现。这对于需要
            在 `if` 语句中使用的上层代码至关重要。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败，或者
            `cupy.isclose` 自身抛出异常，都会被捕获、记录，并重新包装成
            带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            a (Union[complex, float, int, Any]):
                第一个要比较的数值。可以是 `cupy.ndarray` 或 Python 标量。
            b (Union[complex, float, int, Any]):
                第二个要比较的数值。可以是 `cupy.ndarray` 或 Python 标量。
            atol (float, optional):
                绝对容差 (absolute tolerance)。两个数的差的绝对值必须
                小于等于这个值，才被认为是接近的。默认为 `1e-9`。

        Returns:
            bool:
                如果两个数值足够接近，则返回 `True`，否则返回 `False`。

        Raises:
            TypeError: 如果输入 `a`, `b` 无法转换为 CuPy 数组，或 `atol` 不是 `float`。
            ValueError: 如果 `atol` 为负数。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 对 atol 参数进行严格的验证 ---
        if not isinstance(atol, float):
            self._internal_logger.error(f"isclose: Absolute tolerance 'atol' must be a float, but got {type(atol).__name__}.")
            raise TypeError(f"Absolute tolerance 'atol' must be a float.")
        if atol < 0:
            self._internal_logger.error(f"isclose: Absolute tolerance 'atol' cannot be negative, but got {atol}.")
            raise ValueError(f"Absolute tolerance 'atol' cannot be negative.")

        # --- 步骤 2: 将输入安全地转换为 CuPy 数组 ---
        try:
            # 使用 _ensure_cupy_array 来处理 Python 标量和 CuPy 数组
            a_cp = self._ensure_cupy_array(a)
            b_cp = self._ensure_cupy_array(b)
        except TypeError as e:
            self._internal_logger.error(
                f"isclose: One or both inputs could not be converted to a CuPy array. "
                f"Input A type: {type(a).__name__}, Input B type: {type(b).__name__}.",
                exc_info=True
            )

            raise TypeError(
                f"Inputs to isclose must be numeric or CuPy arrays."
            ) from e

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # cupy.isclose 返回一个布尔类型的 CuPy 数组。对于标量输入，这将是一个 0 维数组。
            # rtol=0 表示我们只关心绝对容差，这与 PurePythonBackend 的实现一致。
            close_result_array = self._cp.isclose(a_cp, b_cp, rtol=0, atol=atol)
            
            # --- 步骤 4: 确保返回类型是 Python 的 bool 标量 ---
            scalar_result = self._scalar_or_array(close_result_array)

            # 最终强制转换为 bool，以防 .item() 返回 cupy.bool_ 等特殊类型
            return bool(scalar_result)

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.isclose. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute isclose.") from e

    def allclose(self, mat_a: Any, mat_b: Any, atol: float = 1e-9) -> bool:
        """
        [内部辅助方法] [最终修正增强版] 判断两个类数组对象中的所有对应元素
        是否都在指定的绝对容差内“足够接近”。

        此方法是对 `cupy.allclose` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.allclose` 之前，会通过
            `self._ensure_cupy_array` 方法将输入 `mat_a` 和 `mat_b` 安全地
            转换为 CuPy 数组。这使得此方法能无缝处理 Python 列表、嵌套列表等
            多种输入格式。
        -   **严格的容差验证**: 对 `atol` 参数的类型和值范围进行严格检查，
            确保它是一个非负的浮点数。
        -   **返回类型保证**: 确保返回值总是一个 Python 的 `bool` 标量，
            通过调用 `_scalar_or_array` 和 `bool()` 来实现。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败（例如，
            因为输入的 Python 列表形状不规则），或者 `cupy.allclose` 因形状
            不匹配而抛出异常，都会被捕获、记录，并重新包装成带有清晰上下文
            的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            mat_a (Any):
                第一个要比较的数组。可以是 `cupy.ndarray` 或 Python 列表/嵌套列表。
            mat_b (Any):
                第二个要比较的数组。可以是 `cupy.ndarray` 或 Python 列表/嵌套列表。
            atol (float, optional):
                绝对容差 (absolute tolerance)。默认为 `1e-9`。

        Returns:
            bool:
                如果两个数组形状相同且所有对应元素都足够接近，则返回 `True`，
                否则返回 `False`。

        Raises:
            TypeError: 如果输入 `mat_a` 或 `mat_b` 无法转换为 CuPy 数组，或 `atol` 不是 `float`。
            ValueError: 如果 `atol` 为负数。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 对 atol 参数进行严格的验证 ---
        if not isinstance(atol, float):
            self._internal_logger.error(f"allclose: Absolute tolerance 'atol' must be a float, but got {type(atol).__name__}.")
            raise TypeError(f"Absolute tolerance 'atol' must be a float.")
        if atol < 0:
            self._internal_logger.error(f"allclose: Absolute tolerance 'atol' cannot be negative, but got {atol}.")
            raise ValueError(f"Absolute tolerance 'atol' cannot be negative.")

        # --- 步骤 2: 将输入安全地转换为 CuPy 数组 ---
        try:
            a_cp = self._ensure_cupy_array(mat_a)
            b_cp = self._ensure_cupy_array(mat_b)
        except TypeError as e:
            self._internal_logger.error(
                f"allclose: One or both inputs could not be converted to a CuPy array. "
                f"Input A type: {type(mat_a).__name__}, Input B type: {type(mat_b).__name__}.",
                exc_info=True
            )
            raise TypeError(
                f"Inputs to allclose must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # cupy.allclose 会自动处理形状检查。如果形状不匹配，它会返回 False 或抛出 ValueError，
            # 这取决于 CuPy 的版本和具体情况。我们的 except 块可以处理这些。
            # rtol=0 表示我们只关心绝对容差，这与 PurePythonBackend 的实现一致。
            close_result_array = self._cp.allclose(a_cp, b_cp, rtol=0, atol=atol)
            
            # --- 步骤 4: 确保返回类型是 Python 的 bool 标量 ---
            # cupy.allclose 返回一个 0 维的布尔数组
            scalar_result = self._scalar_or_array(close_result_array)

            # 最终强制转换为 bool
            return bool(scalar_result)

        except ValueError as e:
            # 捕获 CuPy 可能因形状不匹配等原因抛出的 ValueError
            self._internal_logger.warning(
                f"cupy.allclose raised a ValueError, likely due to shape mismatch. "
                f"Shape A: {a_cp.shape}, Shape B: {b_cp.shape}. CuPy Error: {e}"
            )
            # 对于 allclose，形状不匹配意味着它们不“接近”，所以安全地返回 False
            return False
        
        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何其他意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.allclose. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute allclose.") from e

    # --- 数学函数：直接通过CuPy的math模块或NumPy包装 ---
    def sin(self, x: Union[float, int, complex, Any]) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个数值或一个类数组对象中
        所有元素的正弦值。

        此方法是对 `cupy.sin` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.sin` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `x` 安全地转换为 CuPy 数组。
            这使得此方法能无缝处理 Python 标量 (`int`, `float`, `complex`)
            和 CuPy 数组。
        -   **返回类型智能处理**:
            -   如果输入是 Python 标量，则返回 Python 标量（`float` 或 `complex`）。
            -   如果输入是数组，则返回 CuPy 数组。
            这是通过检查原始输入类型并结合 `_scalar_or_array` 实现的，
            提供了更符合用户直觉的 API 行为。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败，或者
            `cupy.sin` 自身抛出异常，都会被捕获、记录，并重新包装成带有
            清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            x (Union[float, int, complex, Any]):
                要计算正弦值的输入。可以是 `cupy.ndarray` 或 Python 标量。

        Returns:
            Any:
                如果输入是标量，则返回计算结果的 `float` 或 `complex` 标量。
                如果输入是数组，则返回一个 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果输入 `x` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 记录原始输入类型 ---
        # 用于后续决定返回类型是标量还是数组
        is_scalar_input = isinstance(x, (int, float, complex))

        # --- 步骤 2: 将输入安全地转换为 CuPy 数组 ---
        try:
            # 这里的 dtype=None 允许 CuPy 自动推断最合适的数据类型
            # （例如，如果输入是 float，则创建 float 数组），
            # 因为 sin 函数可以处理实数和复数。
            x_cp = self._ensure_cupy_array(x, dtype=None)
        except TypeError as e:
            self._internal_logger.error(
                f"sin: The input of type {type(x).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for sin must be numeric or a CuPy array."
            ) from e

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.sin 进行逐元素计算
            result_array = self._cp.sin(x_cp)
            
            # --- 步骤 4: 根据原始输入类型处理返回类型 ---
            if is_scalar_input:
                # 如果原始输入是标量，我们期望返回一个标量。
                # _scalar_or_array 会将 0 维数组转换为 Python 标量。
                return self._scalar_or_array(result_array)
            else:
                # 如果原始输入是数组，则返回 CuPy 数组
                return result_array

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.sin. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute sin.") from e

    def cos(self, x: Union[float, int, complex, Any]) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个数值或一个类数组对象中
        所有元素的余弦值。

        此方法是对 `cupy.cos` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.cos` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `x` 安全地转换为 CuPy 数组。
            这使得此方法能无缝处理 Python 标量 (`int`, `float`, `complex`)
            和 CuPy 数组。
        -   **返回类型智能处理**:
            -   如果输入是 Python 标量，则返回 Python 标量（`float` 或 `complex`）。
            -   如果输入是数组，则返回 CuPy 数组。
            这是通过检查原始输入类型并结合 `_scalar_or_array` 实现的。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败，或者
            `cupy.cos` 自身抛出异常，都会被捕获、记录，并重新包装成带有
            清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            x (Union[float, int, complex, Any]):
                要计算余弦值的输入。可以是 `cupy.ndarray` 或 Python 标量。

        Returns:
            Any:
                如果输入是标量，则返回计算结果的 `float` 或 `complex` 标量。
                如果输入是数组，则返回一个 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果输入 `x` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 记录原始输入类型 ---
        # 用于后续决定返回类型是标量还是数组
        is_scalar_input = isinstance(x, (int, float, complex))

        # --- 步骤 2: 将输入安全地转换为 CuPy 数组 ---
        try:
            # dtype=None 允许 CuPy 自动推断最合适的数据类型
            x_cp = self._ensure_cupy_array(x, dtype=None)
        except TypeError as e:
            self._internal_logger.error(
                f"cos: The input of type {type(x).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for cos must be numeric or a CuPy array."
            ) from e

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.cos 进行逐元素计算
            result_array = self._cp.cos(x_cp)
            
            # --- 步骤 4: 根据原始输入类型处理返回类型 ---
            if is_scalar_input:
                # 如果原始输入是标量，我们期望返回一个标量。
                return self._scalar_or_array(result_array)
            else:
                # 如果原始输入是数组，则返回 CuPy 数组
                return result_array

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.cos. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute cos.") from e

    def exp(self, x: Union[float, int, complex, Any]) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个数值或一个类数组对象中
        所有元素的指数 (e^x)。

        此方法是对 `cupy.exp` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.exp` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `x` 安全地转换为 CuPy 数组。
            这使得此方法能无缝处理 Python 标量 (`int`, `float`, `complex`)
            和 CuPy 数组。
        -   **返回类型智能处理**:
            -   如果输入是 Python 标量，则返回 Python 标量（`float` 或 `complex`）。
            -   如果输入是数组，则返回 CuPy 数组。
            这是通过检查原始输入类型并结合 `_scalar_or_array` 实现的。
        -   **清晰的错误处理**: 如果 `_ensure_cupy_array` 转换失败，或者
            `cupy.exp` 自身抛出异常，都会被捕获、记录，并重新包装成带有
            清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            x (Union[float, int, complex, Any]):
                要计算指数的输入。可以是 `cupy.ndarray` 或 Python 标量。

        Returns:
            Any:
                如果输入是标量，则返回计算结果的 `float` 或 `complex` 标量。
                如果输入是数组，则返回一个 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果输入 `x` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 记录原始输入类型 ---
        # 用于后续决定返回类型是标量还是数组
        is_scalar_input = isinstance(x, (int, float, complex))

        # --- 步骤 2: 将输入安全地转换为 CuPy 数组 ---
        try:
            # dtype=None 允许 CuPy 自动推断最合适的数据类型
            x_cp = self._ensure_cupy_array(x, dtype=None)
        except TypeError as e:
            self._internal_logger.error(
                f"exp: The input of type {type(x).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for exp must be numeric or a CuPy array."
            ) from e

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.exp 进行逐元素计算
            result_array = self._cp.exp(x_cp)
            
            # --- 步骤 4: 根据原始输入类型处理返回类型 ---
            if is_scalar_input:
                # 如果原始输入是标量，我们期望返回一个标量。
                return self._scalar_or_array(result_array)
            else:
                # 如果原始输入是数组，则返回 CuPy 数组
                return result_array

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.exp. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute exp.") from e

    def log2(self, x: Union[float, int, Any]) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个实数或一个类数组对象中
        所有元素的以 2 为底的对数 (log₂x)。

        此方法是对 `cupy.log2` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **定义域验证**: 在调用 `cupy.log2` 之前，会进行检查以确保所有输入
            值都是正数。如果输入是标量且非正，会立即抛出 `ValueError`。
            （注意：对于数组输入，CuPy 会自行处理非正值，通常返回 `nan` 或 `-inf`，
            此行为被保留）。
        -   **灵活的输入处理**: 在计算之前，会通过 `self._ensure_cupy_array`
            方法将输入 `x` 安全地转换为 `float` 类型的 CuPy 数组，因为 `log2`
            只对实数有定义。
        -   **返回类型智能处理**:
            -   如果输入是 Python 标量，则返回 Python `float` 标量。
            -   如果输入是数组，则返回 `float` 类型的 CuPy 数组。
        -   **清晰的错误处理**: 如果输入无法转换或计算失败，都会被捕获、
            记录，并重新包装成带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和数学上的限制（输入必须为正数）。

        Args:
            x (Union[float, int, Any]):
                要计算对数的输入。可以是 `cupy.ndarray` 或 Python 实数标量。
                输入值必须为正数。

        Returns:
            Any:
                如果输入是标量，则返回计算结果的 `float` 标量。
                如果输入是数组，则返回一个 `float` 类型的 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果输入 `x` 无法被转换为一个有效的 CuPy 数组。
            ValueError: 如果输入的标量 `x` 不是正数。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 记录原始输入类型并进行定义域验证 (针对标量) ---
        is_scalar_input = isinstance(x, (int, float))

        if is_scalar_input:
            if x <= 0:
                self._internal_logger.error(f"log2: Input must be positive, but got scalar value {x}.")
                raise ValueError("Input for log2 must be positive.")
        
        # --- 步骤 2: 将输入安全地转换为 float 类型的 CuPy 数组 ---
        try:
            # log2 只对实数定义，因此强制 dtype=float
            x_cp = self._ensure_cupy_array(x, dtype=float)
        except TypeError as e:
            self._internal_logger.error(
                f"log2: The input of type {type(x).__name__} could not be converted to a float CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for log2 must be a real number or an array of real numbers."
            ) from e

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.log2 进行逐元素计算
            result_array = self._cp.log2(x_cp)
            
            # --- 步骤 4: 根据原始输入类型处理返回类型 ---
            if is_scalar_input:
                # 如果原始输入是标量，我们期望返回一个标量。
                return self._scalar_or_array(result_array)
            else:
                # 如果原始输入是数组，则返回 CuPy 数组
                return result_array

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.log2. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute log2.") from e

    def abs(self, x: Union[float, int, complex, Any]) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个数值或一个类数组对象中
        所有元素的绝对值（对于实数）或模（对于复数）。

        此方法是对 `cupy.abs` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.abs` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `x` 安全地转换为 CuPy 数组。
            这使得此方法能无缝处理 Python 标量 (`int`, `float`, `complex`)
            和 CuPy 数组。
        -   **返回类型智能处理**:
            -   如果输入是 Python 标量，则返回 Python `float` 标量。
            -   如果输入是数组，则返回 `float` 类型的 CuPy 数组。
            这是通过检查原始输入类型并结合 `_scalar_or_array` 实现的。
        -   **输出 `dtype` 保证**: 无论输入是 `float` 还是 `complex`，
            `cupy.abs` 的结果都是 `float` 类型的。此方法确保了这一点，
            并相应地处理返回类型。
        -   **清晰的错误处理**: 如果输入无法转换或计算失败，都会被捕获、
            记录，并重新包装成带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            x (Union[float, int, complex, Any]):
                要计算绝对值/模的输入。可以是 `cupy.ndarray` 或 Python 标量。

        Returns:
            Any:
                如果输入是标量，则返回计算结果的 `float` 标量。
                如果输入是数组，则返回一个 `float` 类型的 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果输入 `x` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 记录原始输入类型 ---
        # 用于后续决定返回类型是标量还是数组
        is_scalar_input = isinstance(x, (int, float, complex))

        # --- 步骤 2: 将输入安全地转换为 CuPy 数组 ---
        try:
            # dtype=None 允许 CuPy 自动推断最合适的数据类型
            x_cp = self._ensure_cupy_array(x, dtype=None)
        except TypeError as e:
            self._internal_logger.error(
                f"abs: The input of type {type(x).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for abs must be numeric or a CuPy array."
            ) from e

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.abs 进行逐元素计算。
            # 无论输入是 float 还是 complex，CuPy 都会返回一个 float 类型的数组。
            result_array = self._cp.abs(x_cp)
            
            # --- 步骤 4: 根据原始输入类型处理返回类型 ---
            if is_scalar_input:
                # 如果原始输入是标量，我们期望返回一个标量。
                scalar_result = self._scalar_or_array(result_array)
                # 确保返回的是 Python float
                return float(scalar_result)
            else:
                # 如果原始输入是数组，则返回 CuPy 数组
                return result_array

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.abs. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute abs.") from e

    def sqrt(self, x: Union[float, int, complex, Any]) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个数值或一个类数组对象中
        所有元素的平方根。

        此方法是对 `cupy.sqrt` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口，特别是在处理负实数输入时
        能正确返回复数结果。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.sqrt` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `x` 安全地转换为 CuPy 数组。
            这使得此方法能无缝处理 Python 标量 (`int`, `float`, `complex`)
            和 CuPy 数组。
        -   **正确的类型提升**: 在将输入转换为 CuPy 数组时，如果输入是
            Python 标量，会检查其是否为负实数。如果是，则强制将数组的
            `dtype` 提升为 `complex`，以确保 `cupy.sqrt` 能够正确计算出
            复数结果（例如 `sqrt(-1)` -> `1j`），而不是返回 `nan`。
        -   **返回类型智能处理**:
            -   如果输入是 Python 标量，则返回 Python 标量（`float` 或 `complex`）。
            -   如果输入是数组，则返回 CuPy 数组。
        -   **清晰的错误处理**: 如果输入无法转换或计算失败，都会被捕获、
            记录，并重新包装成带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            x (Union[float, int, complex, Any]):
                要计算平方根的输入。可以是 `cupy.ndarray` 或 Python 标量。

        Returns:
            Any:
                如果输入是标量，则返回计算结果的 `float` 或 `complex` 标量。
                如果输入是数组，则返回一个 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果输入 `x` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 记录原始输入类型并进行类型提升检查 ---
        is_scalar_input = isinstance(x, (int, float, complex))
        
        # 默认让 CuPy 自动推断 dtype
        target_dtype = None
        
        # [核心] 如果输入是负的实数标量，则必须将目标 dtype 提升为 complex
        # 否则 cupy.sqrt(-1) 会返回 nan 而不是 1j。
        if is_scalar_input and not isinstance(x, complex) and x < 0:
            target_dtype = complex

        # --- 步骤 2: 将输入安全地转换为 CuPy 数组 ---
        try:
            x_cp = self._ensure_cupy_array(x, dtype=target_dtype)
        except TypeError as e:
            self._internal_logger.error(
                f"sqrt: The input of type {type(x).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for sqrt must be numeric or a CuPy array."
            ) from e

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.sqrt 进行逐元素计算
            result_array = self._cp.sqrt(x_cp)
            
            # --- 步骤 4: 根据原始输入类型处理返回类型 ---
            if is_scalar_input:
                # 如果原始输入是标量，我们期望返回一个标量。
                return self._scalar_or_array(result_array)
            else:
                # 如果原始输入是数组，则返回 CuPy 数组
                return result_array

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.sqrt. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute sqrt.") from e
    
    # --- 随机数生成 ---
    
    def choice(self, options: List[Any], p: List[float]) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 根据给定的概率分布 `p`，从
        `options` 列表中随机选择一个元素。

        此方法主要用于模拟量子测量。它旨在提供一个与 `PurePythonBackend`
        中对应方法一致的接口。

        核心增强功能:
        -   **严格的输入验证**: 在执行任何操作之前，对 `options` 和 `p`
            的类型、长度、以及 `p` 中概率值的有效性（非负）进行严格检查。
        -   **优先使用 `cupy.random.choice`**: 尝试使用 CuPy 的内置随机选择
            函数，这可能利用 GPU 进行高效的随机数生成（尽管对于单次选择，
            性能差异可能不大）。
        -   **健壮的回退机制 (Fallback)**: `cupy.random.choice` 在某些版本或
            情况下可能不支持从任意对象列表（`List[Any]`）中进行选择。
            因此，此方法实现了一个安全的回退：如果 `cupy.random.choice`
            失败，它会无缝地切换到一个纯 Python 的轮盘赌选择算法，确保
            功能的正确性，而不是让程序崩溃。
        -   **概率归一化**: 在调用随机选择函数之前，会对概率分布 `p` 进行
            归一化，以处理因浮点误差导致总和不完全等于 1 的情况。
        -   **详细的文档**: 文档字符串清晰地说明了函数的双路径行为、参数、
            返回值和可能引发的异常。

        Args:
            options (List[Any]):
                一个包含待选元素的列表。
            p (List[float]):
                一个与 `options` 列表等长的概率分布列表。

        Returns:
            Any:
                从 `options` 中根据概率 `p` 随机选出的一个元素。

        Raises:
            TypeError: 如果 `options` 或 `p` 不是列表，或 `p` 包含非数值元素。
            ValueError: 如果 `options` 和 `p` 长度不匹配，或 `p` 包含负数。
        """
        # --- 步骤 1: 对输入参数进行严格的验证 ---
        if not isinstance(options, list):
            self._internal_logger.error(f"choice: Input 'options' must be a list, but got {type(options).__name__}.")
            raise TypeError("Input 'options' must be a list.")
        if not isinstance(p, list):
            self._internal_logger.error(f"choice: Input 'p' (probabilities) must be a list, but got {type(p).__name__}.")
            raise TypeError("Input 'p' (probabilities) must be a list.")
        
        if len(options) != len(p):
            self._internal_logger.error(f"choice: Length of 'options' ({len(options)}) and 'p' ({len(p)}) must be the same.")
            raise ValueError(f"Length of 'options' ({len(options)}) and 'p' ({len(p)}) must be the same.")
        
        if not options:
            self._internal_logger.warning("choice: 'options' list is empty. Cannot make a choice. Returning None.")
            return None
            
        if any(not isinstance(prob, (float, int)) or prob < 0 for prob in p):
            self._internal_logger.error(f"choice: Probabilities in 'p' must be non-negative numeric values.")
            raise ValueError("All probabilities in 'p' must be non-negative numeric values.")

        # --- 步骤 2: 概率归一化 ---
        prob_sum = sum(p)
        
        if prob_sum < 1e-12: # 如果总概率接近于零
            self._internal_logger.warning("choice: Sum of probabilities is close to zero. Choosing uniformly at random from options as a fallback.")
            # 使用 Python 的 `random.choice` 进行均匀随机选择
            return self.random.choice(options)
        
        normalized_p = [float(val) / prob_sum for val in p]

        # --- 步骤 3: 尝试使用 cupy.random.choice (优先路径) ---
        try:
            # cupy.random.choice 可以接受一个 cupy 数组作为概率分布
            p_cp = self._cp.array(normalized_p, dtype=float)
            
            # cupy.random.choice 返回一个数组，即使只选一个。我们需要取出第一个元素。
            # 注意：cupy.random.choice 在某些版本中可能对 `options` 的类型有限制。
            # 我们的 try...except 结构可以处理这种情况。
            chosen_index = self._cp.random.choice(len(options), p=p_cp)
            
            # .item() 将 0 维 CuPy 数组转换为 Python 标量
            return options[int(chosen_index.item())]

        except Exception as e:
            # --- 步骤 4: 如果 CuPy 路径失败，则回退到纯 Python 实现 ---
            self._internal_logger.warning(
                f"cupy.random.choice failed with error: {e}. "
                "This can happen if the CuPy version has limitations. "
                "Falling back to a pure Python implementation for this choice."
            )

            # 使用 Python 标准库的 random 和轮盘赌选择算法
            r = self.random.random()
            cumulative_prob = 0.0
            for i, prob in enumerate(normalized_p):
                cumulative_prob += prob
                if r < cumulative_prob:
                    return options[i]
            
            # 由于浮点误差，循环可能无法提前返回，此时返回最后一个元素是安全的
            return options[-1]

    def random_normal(self, loc: float = 0.0, scale: float = 1.0, size: Tuple[int, ...]= (1,)) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 生成服从正态（高斯）分布的随机数。

        此方法是对 `cupy.random.normal` 函数的一个健 robust封装，旨在提供一个
        与 `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **严格的输入验证**: 在调用底层的 `cupy.random.normal` 之前，
            对所有输入参数 (`loc`, `scale`, `size`) 的类型和值范围进行
            严格检查。
            -   确保 `loc` 和 `scale` 是实数。
            -   确保 `scale` (标准差) 是非负的。
            -   确保 `size` 是一个包含非负整数的元组。
        -   **返回类型智能处理**:
            -   如果 `size` 是 `(1,)` 或 `()`（表示请求单个标量），则返回
                Python `float` 标量。
            -   如果 `size` 是其他形状，则返回相应形状的 CuPy 数组。
            这是通过检查 `size` 参数并结合 `_scalar_or_array` 实现的。
        -   **清晰的错误处理**: 如果参数验证失败，或者 `cupy.random.normal`
            自身抛出异常，都会被捕获、记录，并重新包装成带有清晰上下文的
            异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            loc (float, optional):
                正态分布的均值 (μ)。默认为 0.0。
            scale (float, optional):
                正态分布的标准差 (σ)。必须是非负的。默认为 1.0。
            size (Tuple[int, ...], optional):
                输出数组的形状。默认为 `(1,)`，表示生成单个标量。

        Returns:
            Any:
                如果请求的是单个值，则返回一个 `float` 标量。
                否则，返回一个 `float` 类型的 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果 `loc`, `scale` 不是实数，或 `size` 的格式不正确。
            ValueError: 如果 `scale` 为负数，或 `size` 包含负数维度。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 对输入参数进行严格的验证 ---
        if not isinstance(loc, (float, int)):
            self._internal_logger.error(f"random_normal: Mean 'loc' must be a real number, but got {type(loc).__name__}.")
            raise TypeError(f"Mean 'loc' must be a real number.")
        
        if not isinstance(scale, (float, int)):
            self._internal_logger.error(f"random_normal: Standard deviation 'scale' must be a real number, but got {type(scale).__name__}.")
            raise TypeError(f"Standard deviation 'scale' must be a real number.")
        
        if scale < 0:
            self._internal_logger.error(f"random_normal: Standard deviation 'scale' cannot be negative, but got {scale}.")
            raise ValueError(f"Standard deviation 'scale' cannot be negative.")
        
        if not isinstance(size, tuple):
             self._internal_logger.error(f"random_normal: Shape 'size' must be a tuple, but got {type(size).__name__}.")
             raise TypeError(f"Shape 'size' must be a tuple.")
        
        for dimension in size:
            if not isinstance(dimension, int):
                self._internal_logger.error(f"random_normal: All elements in 'size' tuple must be integers, but found '{dimension}' of type {type(dimension).__name__} in {size}.")
                raise TypeError(f"All elements in 'size' tuple must be integers.")
            if dimension < 0:
                self._internal_logger.error(f"random_normal: Dimensions in 'size' tuple cannot be negative, but found '{dimension}' in {size}.")
                raise ValueError(f"Dimensions in 'size' tuple cannot be negative.")

        # --- 步骤 2: 记录是否请求了标量输出 ---
        is_scalar_output_requested = (size == (1,) or size == ())
        
        # CuPy 的 size 参数对于单个值可以是 None 或 1，但元组是标准形式
        size_for_cupy = size if size != () else None

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.random.normal。明确指定 dtype=float。
            result_array = self._cp.random.normal(loc=loc, scale=scale, size=size_for_cupy, dtype=float)
            
            # --- 步骤 4: 根据请求处理返回类型 ---
            if is_scalar_output_requested:
                # 如果请求的是单个值，我们期望返回一个 Python float。
                return self._scalar_or_array(result_array)
            else:
                # 如果请求的是数组，则返回 CuPy 数组
                return result_array

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.random.normal. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to generate random normal numbers.") from e

    def clip(self, values: Any, min_val: float, max_val: float) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 将一个数值或一个类数组对象中的所有
        数值裁剪到指定的 `[min_val, max_val]` 区间内。

        此方法是对 `cupy.clip` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **严格的输入验证**: 在调用底层的 `cupy.clip` 之前，对所有输入
            参数 (`values`, `min_val`, `max_val`) 的类型和值范围进行严格检查。
            -   确保 `min_val` 和 `max_val` 是实数，且 `min_val <= max_val`。
        -   **灵活的输入处理**: 在计算之前，会通过 `self._ensure_cupy_array`
            方法将输入 `values` 安全地转换为 CuPy 数组。这使得此方法能无缝
            处理 Python 标量和数组。
        -   **返回类型智能处理**:
            -   如果输入 `values` 是 Python 标量，则返回 Python 标量。
            -   如果输入是数组，则返回 CuPy 数组。
        -   **清晰的错误处理**: 如果参数验证失败，或 `cupy.clip` 自身抛出
            异常，都会被捕获、记录，并重新包装成带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            values (Any):
                要裁剪的输入。可以是 `cupy.ndarray` 或 Python 标量。
            min_val (float):
                区间的下界。
            max_val (float):
                区间的上界。

        Returns:
            Any:
                如果输入是标量，则返回裁剪后的标量。
                如果输入是数组，则返回一个裁剪后的 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果输入 `values` 无法转换为 CuPy 数组，或 `min_val`,
                       `max_val` 不是实数。
            ValueError: 如果 `min_val > max_val`。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 对边界参数进行严格的验证 ---
        if not isinstance(min_val, (float, int)):
            self._internal_logger.error(f"clip: Boundary 'min_val' must be a numeric type, but got {type(min_val).__name__}.")
            raise TypeError(f"Boundary 'min_val' must be a numeric type.")
        
        if not isinstance(max_val, (float, int)):
            self._internal_logger.error(f"clip: Boundary 'max_val' must be a numeric type, but got {type(max_val).__name__}.")
            raise TypeError(f"Boundary 'max_val' must be a numeric type.")
        
        if min_val > max_val:
            self._internal_logger.error(f"clip: The minimum value ({min_val}) cannot be greater than the maximum value ({max_val}).")
            raise ValueError(f"The minimum value ({min_val}) cannot be greater than the maximum value ({max_val}).")

        # --- 步骤 2: 记录原始输入类型 ---
        is_scalar_input = isinstance(values, (int, float, complex))

        # --- 步骤 3: 将输入安全地转换为 CuPy 数组 ---
        try:
            # dtype=None 允许 CuPy 自动推断
            values_cp = self._ensure_cupy_array(values, dtype=None)
        except TypeError as e:
            self._internal_logger.error(
                f"clip: The input 'values' of type {type(values).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input 'values' for clip must be numeric or a CuPy array."
            ) from e

        # --- 步骤 4: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.clip 进行逐元素裁剪
            result_array = self._cp.clip(values_cp, min_val, max_val)
            
            # --- 步骤 5: 根据原始输入类型处理返回类型 ---
            if is_scalar_input:
                # 如果原始输入是标量，我们期望返回一个标量。
                return self._scalar_or_array(result_array)
            else:
                # 如果原始输入是数组，则返回 CuPy 数组
                return result_array

        except Exception as e:
            # --- 步骤 6: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.clip. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute clip.") from e

    def arange(self, dim: int, dtype: Any = None) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 创建一个包含从 0 到 `dim-1` 的
        整数序列的 CuPy 数组。

        此方法是对 `cupy.arange` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **严格的输入验证**: 在调用底层的 `cupy.arange` 之前，对 `dim` 参数
            的类型和值范围进行严格检查，确保它是一个非负整数。
        -   **清晰的错误处理**: 当验证失败时，会记录详细的 `ERROR` 级别日志，
            并抛出明确的 `ValueError` 或 `TypeError`，立即中止不正确的
            操作。如果 `cupy.arange` 自身抛出异常，也会被捕获并重新包装。
        -   **灵活的 `dtype` 处理**: `dtype` 参数被正确地传递给底层的 `cupy.arange`
            函数，允许调用者根据需要指定输出数组的数据类型。如果为 `None`，
            则使用 CuPy 的默认类型（通常是 `int`）。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            dim (int):
                序列的结束值（不包含）。数组将包含 `[0, 1, ..., dim-1]`。
                必须是非负整数。
            dtype (Any, optional):
                输出数组的期望数据类型。如果为 `None`，则使用 CuPy 的默认值。
                默认为 `None`。

        Returns:
            Any:
                一个一维的 `cupy.ndarray` 对象，包含了指定的整数序列。

        Raises:
            TypeError: 如果 `dim` 不是整数。
            ValueError: 如果 `dim` 为负数。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 对输入参数进行严格的验证 ---

        # 验证 'dim' 的类型
        if not isinstance(dim, int):
            self._internal_logger.error(f"arange: Dimension 'dim' must be an integer, but got {type(dim).__name__}.")
            raise TypeError("Dimension 'dim' for arange must be an integer.")
        
        # 验证 'dim' 的值范围
        if dim < 0:
            self._internal_logger.error(f"arange: Dimension 'dim' cannot be negative, but got {dim}.")
            raise ValueError("Dimension 'dim' for arange cannot be negative.")

        # --- 步骤 2: 调用底层的 CuPy 函数来创建数组 ---
        try:
            # 使用 cupy.arange 函数创建序列数组。
            # 将 dtype 参数直接传递给底层函数。
            return self._cp.arange(dim, dtype=dtype)
        
        except Exception as e:
            # --- 步骤 3: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.arange for dimension {dim}. Error: {e}",
                exc_info=True
            )
            # 将底层错误包装成一个更通用的 RuntimeError，向上层报告。
            raise RuntimeError(f"Failed to create CuPy arange array for dimension {dim}.") from e
        
    def power(self, base: Union[complex, float, int, Any], exp: Union[complex, float, int, Any]) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算 `base` 的 `exp` 次幂 (`base ** exp`)，
        支持标量和类数组对象。

        此方法是对 `cupy.power` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.power` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `base` 和 `exp` 安全地
            转换为 CuPy 数组。这使得此方法能无缝处理 Python 标量和数组的
            各种组合（标量**标量, 数组**标量, 标量**数组, 数组**数组）。
        -   **返回类型智能处理**:
            -   如果 `base` 和 `exp` 都是 Python 标量，则返回 Python 标量。
            -   如果至少有一个输入是数组，则返回 CuPy 数组。
            这提供了更符合用户直觉的 API 行为。
        -   **清晰的错误处理**: 如果输入无法转换，或 `cupy.power` 因形状不兼容
            （广播失败）等原因抛出异常，都会被捕获、记录，并重新包装成带有
            清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            base (Union[complex, float, int, Any]):
                底数。可以是 `cupy.ndarray` 或 Python 标量。
            exp (Union[complex, float, int, Any]):
                指数。可以是 `cupy.ndarray` 或 Python 标量。

        Returns:
            Any:
                如果两个输入都是标量，则返回计算结果的标量。
                否则，返回一个 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果输入 `base` 或 `exp` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 记录原始输入类型 ---
        is_scalar_input = isinstance(base, (int, float, complex)) and \
                          isinstance(exp, (int, float, complex))

        # --- 步骤 2: 将输入安全地转换为 CuPy 数组 ---
        try:
            # dtype=None 允许 CuPy 自动推断最合适的数据类型，支持复数运算
            base_cp = self._ensure_cupy_array(base, dtype=None)
            exp_cp = self._ensure_cupy_array(exp, dtype=None)
        except TypeError as e:
            self._internal_logger.error(
                f"power: One or both inputs could not be converted to a CuPy array. "
                f"Base type: {type(base).__name__}, Exp type: {type(exp).__name__}.",
                exc_info=True
            )
            raise TypeError(
                f"Inputs for power must be numeric or CuPy arrays."
            ) from e

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.power。CuPy 会自动处理广播 (broadcasting)。
            result_array = self._cp.power(base_cp, exp_cp)
            
            # --- 步骤 4: 根据原始输入类型处理返回类型 ---
            if is_scalar_input:
                # 如果两个输入都是标量，我们期望返回一个标量。
                return self._scalar_or_array(result_array)
            else:
                # 如果至少有一个输入是数组，则返回 CuPy 数组
                return result_array

        except ValueError as e:
            # 捕获 CuPy 可能因广播失败等原因抛出的 ValueError
            self._internal_logger.error(
                f"cupy.power raised a ValueError, likely due to incompatible shapes for broadcasting. "
                f"Base shape: {base_cp.shape}, Exp shape: {exp_cp.shape}. CuPy Error: {e}",
                exc_info=True
            )
            raise ValueError(
                f"Incompatible shapes for power operation: {base_cp.shape} and {exp_cp.shape}"
            ) from e
        
        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何其他意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.power. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute power.") from e

    def sum(self, values: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个类数组对象中所有元素的总和。

        此方法是对 `cupy.sum` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.sum` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `values` 安全地转换为
            CuPy 数组。这使得此方法能无缝处理 Python 列表和嵌套列表。
        -   **返回类型保证**: 确保返回值总是一个 Python 的标量（`int`,
            `float` 或 `complex`），通过调用 `_scalar_or_array` 来实现。
            `cupy.sum` 对整个数组求和的结果是一个 0 维数组，此方法会
            将其正确地“解包”。
        -   **清晰的错误处理**: 如果输入无法转换或计算失败，都会被捕获、
            记录，并重新包装成带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            values (Any):
                要求和的输入。可以是 `cupy.ndarray`、Python 列表或嵌套列表。

        Returns:
            Any:
                一个 Python 标量（`int`, `float` 或 `complex`），代表所有
                元素的总和。

        Raises:
            TypeError: 如果输入 `values` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 CuPy 数组 ---
        try:
            # dtype=None 允许 CuPy 自动推断最合适的数据类型
            values_cp = self._ensure_cupy_array(values, dtype=None)
        except TypeError as e:
            self._internal_logger.error(
                f"sum: The input of type {type(values).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for sum must be a list of numbers or a CuPy array."
            ) from e

        # --- 步骤 2: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.sum 对数组中的所有元素进行求和。
            # 结果将是一个 0 维的 CuPy 数组。
            sum_result_array = self._cp.sum(values_cp)
            
            # --- 步骤 3: 确保返回类型是 Python 标量 ---
            # _scalar_or_array 会将 0 维数组转换为 Python 标量
            return self._scalar_or_array(sum_result_array)

        except Exception as e:
            # --- 步骤 4: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.sum. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute sum.") from e

    # --- [核心修正 1.2：为后端添加 norm_sq 方法] ---
    def norm_sq(self, vec: Any) -> float:
        """计算 CuPy 向量的范数平方。"""
        vec_cp = self._ensure_cupy_array(vec)
        # CuPy/NumPy 中，norm_sq = (vec.conj() * vec).real.sum()
        norm_sq_result = self._cp.sum(self._cp.abs(vec_cp)**2)
        return float(self._scalar_or_array(norm_sq_result))
    def min(self, arr: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个类数组对象中所有元素的最小值。

        此方法是对 `cupy.min` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.min` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `arr` 安全地转换为
            CuPy 数组。这使得此方法能无缝处理 Python 列表和嵌套列表。
        -   **返回类型保证**: 确保返回值总是一个 Python 的标量（`int` 或
            `float`），通过调用 `_scalar_or_array` 来实现。`cupy.min`
            对整个数组求最小值的结果是一个 0 维数组，此方法会将其正确地“解包”。
        -   **处理空数组**: 对输入为空数组的边缘情况进行了处理，会抛出一个
            `ValueError`，与 `cupy.min` 和 `numpy.min` 的标准行为保持一致。
        -   **清晰的错误处理**: 如果输入无法转换或计算失败，都会被捕获、
            记录，并重新包装成带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            arr (Any):
                要计算最小值的输入。可以是 `cupy.ndarray`、Python 列表或嵌套列表。
                数组中的元素应该是可比较的实数。

        Returns:
            Any:
                一个 Python 标量（`int` 或 `float`），代表所有元素的最小值。

        Raises:
            TypeError: 如果输入 `arr` 无法被转换为一个有效的 CuPy 数组。
            ValueError: 如果输入 `arr` 为空。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 CuPy 数组 ---
        try:
            # dtype=None 允许 CuPy 自动推断最合适的数据类型
            arr_cp = self._ensure_cupy_array(arr, dtype=None)
        except TypeError as e:
            self._internal_logger.error(
                f"min: The input of type {type(arr).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for min must be a list of numbers or a CuPy array."
            ) from e

        # --- 步骤 2: 处理空数组的边缘情况 ---
        # cupy.min 在空数组上会抛出 ValueError，我们在此保持一致的行为
        if arr_cp.size == 0:
            self._internal_logger.error("min: Attempt to get minimum value of an empty array.")
            raise ValueError("zero-size array to reduction operation minimum which has no identity")

        # --- 步骤 3: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.min 对数组中的所有元素求最小值。
            # 结果将是一个 0 维的 CuPy 数组。
            min_result_array = self._cp.min(arr_cp)
            
            # --- 步骤 4: 确保返回类型是 Python 标量 ---
            # _scalar_or_array 会将 0 维数组转换为 Python 标量
            return self._scalar_or_array(min_result_array)

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.min. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute min.") from e
        
    def all(self, arr: Any) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 检查一个类数组对象中的所有元素是否都为 True。

        此方法是对 `cupy.all` 函数的一个健 robust封装，旨在提供一个与
        `PurePythonBackend` 中对应方法一致的接口。

        核心增强功能:
        -   **灵活的输入处理**: 在调用底层的 `cupy.all` 之前，会先通过
            `self._ensure_cupy_array` 方法将输入 `arr` 安全地转换为
            `bool` 类型的 CuPy 数组。这使得此方法能无缝处理 Python 列表
            和嵌套列表，并正确地将 Python 的“真值”（truthy）概念
            （如非零数字、非空字符串）转换为布尔值。
        -   **返回类型保证**: 确保返回值总是一个 Python 的 `bool` 标量，
            通过调用 `_scalar_or_array` 和 `bool()` 来实现。`cupy.all`
            对整个数组求值的结果是一个 0 维数组，此方法会将其正确地“解包”。
        -   **处理空数组**: 对输入为空数组的边缘情况进行了处理，会返回 `True`，
            与 `cupy.all` 和 Python 内置 `all()` 对空可迭代对象的行为保持一致。
        -   **清晰的错误处理**: 如果输入无法转换或计算失败，都会被捕获、
            记录，并重新包装成带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、参数、
            返回值和可能引发的异常。

        Args:
            arr (Any):
                要检查的输入。可以是 `cupy.ndarray`、Python 列表或嵌套列表。

        Returns:
            bool:
                如果数组为空或所有元素都为 True，则返回 `True`。
                否则返回 `False`。

        Raises:
            TypeError: 如果输入 `arr` 无法被转换为一个有效的 CuPy 数组。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 bool 类型的 CuPy 数组 ---
        try:
            # 强制 dtype=bool，让 CuPy 处理 Python 对象的“真值”转换
            arr_cp = self._ensure_cupy_array(arr, dtype=bool)
        except TypeError as e:
            self._internal_logger.error(
                f"all: The input of type {type(arr).__name__} could not be converted to a boolean CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for all must be a list or a CuPy array."
            ) from e

        # --- 步骤 2: 调用底层的 CuPy 函数进行计算 ---
        try:
            # 调用 cupy.all 对数组中的所有元素进行逻辑与操作。
            # 对于空数组，cupy.all 会返回 True，这与 Python 的 all([]) 行为一致。
            # 结果将是一个 0 维的 CuPy 布尔数组。
            all_result_array = self._cp.all(arr_cp)
            
            # --- 步骤 3: 确保返回类型是 Python 的 bool 标量 ---
            scalar_result = self._scalar_or_array(all_result_array)

            # 最终强制转换为 bool，以防 .item() 返回 cupy.bool_ 等特殊类型
            return bool(scalar_result)

        except Exception as e:
            # --- 步骤 4: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.all. Error: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute all.") from e

    # --- 线性代数特定函数 ---
    def eigvalsh(self, mat: Any, max_iterations: int = 1000, tolerance: float = 1e-12) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 计算一个厄米（Hermitian）矩阵的特征值。

        此方法是对 `cupy.linalg.eigvalsh` 函数的一个健 robust封装，旨在提供一个
        与 `PurePythonBackend` 中对应方法一致的接口。`eigvalsh` 专门用于
        厄米矩阵，其计算效率通常高于通用的 `eigvals`，并且保证返回的特征值
        都是实数。

        核心增强功能:
        -   **灵活的输入处理**: 在计算之前，会通过 `self._ensure_cupy_array`
            方法将输入 `mat` 安全地转换为 `complex` 类型的 CuPy 数组。
        -   **严格的形状和属性验证**:
            -   确保输入是一个二维方阵。
            -   **增加厄米性检查**: 在调用 `cupy.linalg.eigvalsh` 之前，会
                验证矩阵是否满足厄米条件 (M = M†)，如果不满足，则抛出一个
                明确的 `ValueError`。这可以防止将非厄米矩阵错误地传递给
                一个期望厄米矩阵的函数。
        -   **忽略未使用参数**: 文档明确说明，`max_iterations` 和 `tolerance`
            参数是为了与 `PurePythonBackend` 的接口保持一致而被接受，但在
            CuPy 后端中它们没有作用，因为计算由 CuPy 的底层库（如 cuSOLVER）
            处理。
        -   **清晰的错误处理**: 如果输入无效或计算失败，都会被捕获、记录，并
            重新包装成带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、对输入矩阵的
            厄米性要求、参数和返回值。

        Args:
            mat (Any):
                要计算特征值的厄米矩阵。可以是 `cupy.ndarray` 或 Python 嵌套列表。
            max_iterations (int, optional):
                为了接口兼容性而存在，但在 CuPy 后端中未使用。
            tolerance (float, optional):
                为了接口兼容性而存在，但在 CuPy 后端中未使用。

        Returns:
            Any:
                一个一维的 `cupy.ndarray` 对象，包含了矩阵的所有特征值（实数），
                按升序排列。

        Raises:
            TypeError: 如果输入 `mat` 无法被转换为一个有效的 CuPy 数组。
            ValueError: 如果输入数组不是一个二维方阵，或者不是厄米矩阵。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 complex 类型的 CuPy 数组 ---
        try:
            mat_cp = self._ensure_cupy_array(mat, dtype=complex)
        except TypeError as e:
            self._internal_logger.error(
                f"eigvalsh: The input of type {type(mat).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for eigvalsh must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 2: 验证数组是否为二维方阵 ---
        if not hasattr(mat_cp, 'ndim') or mat_cp.ndim != 2:
            shape_str = str(mat_cp.shape) if hasattr(mat_cp, 'shape') else "unknown"
            self._internal_logger.error(f"eigvalsh: Input must be a 2D matrix, but got an array with {mat_cp.ndim} dimensions and shape {shape_str}.")
            raise ValueError(f"eigvalsh requires a 2D matrix, but got array of shape {shape_str}.")
        
        rows, cols = mat_cp.shape
        if rows != cols:
            self._internal_logger.error(f"eigvalsh: Input matrix must be square, but got a matrix of shape ({rows}, {cols}).")
            raise ValueError(f"eigvalsh requires a square matrix, but got shape ({rows}, {cols}).")
        
        # --- 步骤 3: 验证矩阵是否为厄米矩阵 (M = M†) ---
        if not self._cp.allclose(mat_cp, mat_cp.conj().T, rtol=0, atol=1e-9):
            self._internal_logger.error("eigvalsh: Input matrix is not Hermitian (M != M.conj().T).")
            # 计算一个差异度量以帮助调试
            diff = self._cp.max(self._cp.abs(mat_cp - mat_cp.conj().T))
            self._internal_logger.error(f"Max absolute difference to its conjugate transpose: {diff.item():.2e}")
            raise ValueError("Input matrix for eigvalsh must be Hermitian.")

        # --- 步骤 4: 调用底层的 CuPy 函数进行计算 ---
        try:
            # cupy.linalg.eigvalsh 专门用于厄米矩阵，返回实数特征值，且通常已排序
            return self._cp.linalg.eigvalsh(mat_cp)

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            shape_str = str(mat_cp.shape)
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.linalg.eigvalsh for a matrix of shape {shape_str}. Error: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to compute eigenvalues for a Hermitian matrix of shape {shape_str}.") from e

    def eigh(self, mat: Any, max_iterations: int = 1000, tolerance: float = 1e-12) -> Tuple[Any, Any]:
        """
        [内部辅助方法] [最终修正增强版] 计算一个厄米（Hermitian）矩阵的
        特征值和特征向量。

        此方法是对 `cupy.linalg.eigh` 函数的一个健 robust封装，旨在提供一个
        与 `PurePythonBackend` 中对应方法一致的接口。`eigh` 专门用于
        厄米矩阵，其计算效率通常高于通用的 `eig`，并且保证返回的特征值
        都是实数。

        核心增强功能:
        -   **灵活的输入处理**: 在计算之前，会通过 `self._ensure_cupy_array`
            方法将输入 `mat` 安全地转换为 `complex` 类型的 CuPy 数组。
        -   **严格的形状和属性验证**:
            -   确保输入是一个二维方阵。
            -   **增加厄米性检查**: 在调用 `cupy.linalg.eigh` 之前，会
                验证矩阵是否满足厄米条件 (M = M†)，如果不满足，则抛出一个
                明确的 `ValueError`。
        -   **忽略未使用参数**: 文档明确说明，`max_iterations` 和 `tolerance`
            参数是为了与 `PurePythonBackend` 的接口保持一致而被接受，但在
            CuPy 后端中它们没有作用。
        -   **清晰的错误处理**: 如果输入无效或计算失败，都会被捕获、记录，并
            重新包装成带有清晰上下文的异常向上层抛出。
        -   **详细的文档**: 文档字符串清晰地说明了函数的功能、对输入矩阵的
            厄米性要求、参数和返回值的结构。

        Args:
            mat (Any):
                要计算特征值和特征向量的厄米矩阵。可以是 `cupy.ndarray`
                或 Python 嵌套列表。
            max_iterations (int, optional):
                为了接口兼容性而存在，但在 CuPy 后端中未使用。
            tolerance (float, optional):
                为了接口兼容性而存在，但在 CuPy 后端中未使用。

        Returns:
            Tuple[Any, Any]:
                一个元组 `(eigenvalues, eigenvectors)`:
                - `eigenvalues`: 一个一维的 `cupy.ndarray`，包含了所有特征值（实数），
                  按升序排列。
                - `eigenvectors`: 一个二维的 `cupy.ndarray`，其列是与特征值
                  相对应的归一化特征向量。

        Raises:
            TypeError: 如果输入 `mat` 无法被转换为一个有效的 CuPy 数组。
            ValueError: 如果输入数组不是一个二维方阵，或者不是厄米矩阵。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 将输入安全地转换为 complex 类型的 CuPy 数组 ---
        try:
            mat_cp = self._ensure_cupy_array(mat, dtype=complex)
        except TypeError as e:
            self._internal_logger.error(
                f"eigh: The input of type {type(mat).__name__} could not be converted to a CuPy array.",
                exc_info=True
            )
            raise TypeError(
                f"Input for eigh must be convertible to a CuPy array (e.g., a regular nested list)."
            ) from e

        # --- 步骤 2: 验证数组是否为二维方阵 ---
        if not hasattr(mat_cp, 'ndim') or mat_cp.ndim != 2:
            shape_str = str(mat_cp.shape) if hasattr(mat_cp, 'shape') else "unknown"
            self._internal_logger.error(f"eigh: Input must be a 2D matrix, but got an array with {mat_cp.ndim} dimensions and shape {shape_str}.")
            raise ValueError(f"eigh requires a 2D matrix, but got array of shape {shape_str}.")
        
        rows, cols = mat_cp.shape
        if rows != cols:
            self._internal_logger.error(f"eigh: Input matrix must be square, but got a matrix of shape ({rows}, {cols}).")
            raise ValueError(f"eigh requires a square matrix, but got shape ({rows}, {cols}).")
        
        # --- 步骤 3: 验证矩阵是否为厄米矩阵 (M = M†) ---
        if not self._cp.allclose(mat_cp, mat_cp.conj().T, rtol=0, atol=1e-9):
            self._internal_logger.error("eigh: Input matrix is not Hermitian (M != M.conj().T).")
            # 计算一个差异度量以帮助调试
            diff = self._cp.max(self._cp.abs(mat_cp - mat_cp.conj().T))
            self._internal_logger.error(f"Max absolute difference to its conjugate transpose: {diff.item():.2e}")
            raise ValueError("Input matrix for eigh must be Hermitian.")

        # --- 步骤 4: 调用底层的 CuPy 函数进行计算 ---
        try:
            # cupy.linalg.eigh 专门用于厄米矩阵，返回实数特征值和对应的特征向量
            return self._cp.linalg.eigh(mat_cp)

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装来自 CuPy 的任何意外错误 ---
            shape_str = str(mat_cp.shape)
            self._internal_logger.critical(
                f"An unexpected error occurred in cupy.linalg.eigh for a matrix of shape {shape_str}. Error: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to compute eigenvalues and eigenvectors for a Hermitian matrix of shape {shape_str}.") from e

    def _partial_trace_statevector_kernel(self, vec_tensor: Any, vec_tensor_conj: Any, subscripts: str) -> Any:
        """
        [内部辅助方法] [最终修正增强版] 一个专门为 CuPy 实现的、用于从态矢量
        计算部分迹（约化密度矩阵）的核心函数。

        此内核通过 CuPy 的高级索引和向量化操作，高效地计算 ρ_A = Tr_B(|ψ⟩⟨ψ|)，
        其中 |ψ⟩ 是完整的系统态矢量，A 是要保留的子系统，B 是要迹掉（trace out）
        的子系统。

        核心增强功能:
        -   **严格的输入验证**: 在计算之前，对输入张量 `vec_tensor` 和
            `vec_tensor_conj` 的类型、维度和一致性进行严格检查。
        -   **健壮的下标解析**: 复用了 `PurePythonBackend` 中经过验证的静态方法
            `_parse_einsum_for_partial_trace` 来解析 `subscripts` 字符串。
            这确保了两个后端在解释下标、确定哪些量子比特被保留或迹掉方面的
            逻辑是完全一致的。
        -   **高效的向量化计算**:
            1.  将输入态矢量展平（ravel）以进行快速索引。
            2.  通过位运算和 CuPy 数组操作，向量化地生成所有需要求和的
                全局索引掩码。
            3.  使用高级索引（fancy indexing）一次性地从展平的态矢量中提取
                所有相关的振幅。
            4.  使用 `cupy.dot` 对提取出的振幅向量进行内积，高效地完成求和。
        -   **清晰的错误处理**: 整个计算过程被包裹在 `try...except` 块中，
            任何在解析、索引或计算中发生的错误都会被捕获、记录，并重新包装成
            带有清晰上下文的 `RuntimeError`。
        -   **详细的文档与注释**: 详细的文档字符串和代码内注释解释了算法的
            每一步，包括 `einsum` 字符串的解析逻辑和向量化求和的实现细节。

        Args:
            vec_tensor (Any):
                原始态矢量张量（`cupy.ndarray`）。
            vec_tensor_conj (Any):
                共轭态矢量张量（`cupy.ndarray`）。
            subscripts (str):
                `einsum` 下标字符串 (例如, `"aB,Ab->aA"`)。

        Returns:
            Any:
                约化后的密度矩阵，作为一个二维的 `cupy.ndarray` 对象。

        Raises:
            TypeError: 如果输入张量不是 `cupy.ndarray`。
            ValueError: 如果张量形状不正确或下标解析失败。
            RuntimeError: 如果在 CuPy 内部计算时发生意外错误。
        """
        # --- 步骤 1: 对输入参数进行严格的验证 ---
        cp = self._cp
        
        if not isinstance(vec_tensor, cp.ndarray) or not isinstance(vec_tensor_conj, cp.ndarray):
            self._internal_logger.error(
                f"_partial_trace_statevector_kernel: Input tensors must be CuPy arrays, "
                f"but got {type(vec_tensor).__name__} and {type(vec_tensor_conj).__name__}."
            )
            raise TypeError("Input tensors for _partial_trace_statevector_kernel must be CuPy arrays.")

        if vec_tensor.ndim != vec_tensor_conj.ndim or vec_tensor.shape != vec_tensor_conj.shape:
            self._internal_logger.error(
                f"_partial_trace_statevector_kernel: Input tensors must have the same shape, "
                f"but got {vec_tensor.shape} and {vec_tensor_conj.shape}."
            )
            raise ValueError("Input tensors for _partial_trace_statevector_kernel must have the same shape.")

        num_total_qubits = vec_tensor.ndim
        
        try:
            # --- 步骤 2: 复用 PurePythonBackend 的静态方法来健壮地解析 einsum 字符串 ---
            # 这确保了两个后端对下标的解释逻辑完全一致。
            
            # 为了能够解析 "aB,Ab->aA" 这种格式，我们需要将其转换为 "abAB->aA" 的形式
            input_parts = subscripts.split('->')[0].split(',')
            if len(input_parts) != 2:
                raise ValueError(f"Input subscripts must have exactly two parts separated by ',', but got '{input_parts}'.")
            
            # 这是一个简化的转换，假设大写字母在第二个输入中是唯一的
            combined_input_subs = input_parts[0] + input_parts[1].upper()
            output_subs = subscripts.split('->')[1]
            einsum_str_for_parsing = f"{combined_input_subs}->{output_subs}"
            
            qubits_to_keep, qubits_to_trace = PurePythonBackend._parse_einsum_for_partial_trace(einsum_str_for_parsing, num_total_qubits)
            
            num_qubits_kept = len(qubits_to_keep)
            dim_out = 1 << num_qubits_kept

            # --- 步骤 3: 初始化并准备向量化计算 ---
            reduced_rho_matrix = cp.zeros((dim_out, dim_out), dtype=complex)
            
            # 将输入张量展平，以便使用高级索引
            flat_vec = vec_tensor.ravel()
            flat_vec_conj = vec_tensor_conj.ravel()
            
            # --- 步骤 4: 循环遍历输出密度矩阵的每一个元素 ---
            for i_row_local in range(dim_out):
                for i_col_local in range(dim_out):
                    
                    # --- a) 构建用于筛选态矢量元素的基准掩码 ---
                    # 这个掩码由被保留的比特（子系统 A）的状态决定
                    bra_base_mask = 0
                    ket_base_mask = 0
                    for local_idx, global_q_idx in enumerate(qubits_to_keep):
                        # 注意：qubits_to_keep 是排序的，所以 local_idx 对应于
                        # 约化密度矩阵的低位比特
                        if (i_row_local >> local_idx) & 1:
                            bra_base_mask |= (1 << global_q_idx)
                        if (i_col_local >> local_idx) & 1:
                            ket_base_mask |= (1 << global_q_idx)

                    # --- b) 向量化地构建所有要求和的索引 ---
                    # 构建所有可能的迹掉比特（子系统 B）的组合
                    num_trace_configs = 1 << len(qubits_to_trace)
                    if num_trace_configs > 0:
                        trace_configs_gpu = cp.arange(num_trace_configs, dtype=cp.uint32)
                        trace_masks_gpu = cp.zeros_like(trace_configs_gpu)
                        
                        for local_idx, global_q_idx in enumerate(qubits_to_trace):
                            # 对于每个迹掉的比特，根据 trace_configs_gpu 的相应位，
                            # 将其贡献加到掩码中
                            trace_masks_gpu |= ((trace_configs_gpu >> local_idx) & 1) << global_q_idx

                        # 组合基准掩码和迹掉掩码，得到所有要求和的全局索引
                        global_bra_indices = bra_base_mask | trace_masks_gpu
                        global_ket_indices = ket_base_mask | trace_masks_gpu
                    else: # 如果没有比特被迹掉（即计算完整的密度矩阵）
                        global_bra_indices = cp.array([bra_base_mask], dtype=cp.uint32)
                        global_ket_indices = cp.array([ket_base_mask], dtype=cp.uint32)
                    
                    # --- c) 使用高级索引一次性提取所有相关振幅 ---
                    bra_amps = flat_vec[global_bra_indices]
                    ket_amps_conj = flat_vec_conj[global_ket_indices]
                    
                    # --- d) 执行点积来完成求和: Σ_k (ψ_{ik} * ψ_{jk}^*) ---
                    sum_val = cp.dot(bra_amps, ket_amps_conj)
                    
                    reduced_rho_matrix[i_row_local, i_col_local] = sum_val

            return reduced_rho_matrix

        except Exception as e:
            # --- 步骤 5: 捕获并重新包装任何在内核中发生的错误 ---
            self._internal_logger.critical(
                f"An unexpected error occurred in the specialized _partial_trace_statevector_kernel for subscripts '{subscripts}'. Error: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed during specialized partial trace calculation for '{subscripts}'.") from e

# ========================================================================
# --- 5. [核心] 全局并行计算基础设施 (内化管理版) ---
# ========================================================================

# --- 模块级全局变量，用于管理并行计算状态 ---
# [最终修正版]
# _process_pool 不再被用作持久化池，因此我们将其移除或注释掉，以避免混淆。
# 我们只保留控制并行模式开关、进程数量以及可复用共享队列所需的状态。
_process_pool: Optional[pool.Pool] = None  # 不再作为持久化池使用，仅为类型提示保留或可完全移除
_parallel_enabled: bool = False
_num_processes: int = 0
_pool_lock: mp.Lock = mp.Lock() # 用于保护全局状态的创建/销毁，防止并发问题
_progress_queue: Optional[mp.Queue] = None # 这个队列由一个Manager管理，可以在多次并行调用之间复用

# --- [关键] 进程级全局变量，用于在子进程中存储共享内存引用 ---
# 这个变量将在每个工作进程启动时通过 `init_worker` 函数被赋值。
# 它的作用域是每个独立的子进程，而不是主进程。
global_shm_arrays: Optional[Dict[str, RawArray]] = None

def init_worker(log_level: int, log_format: str, queue: Optional[mp.Queue], shm_arrays_from_main: Dict[str, RawArray]):
    """
    [最终修正增强版] 每个并行 worker 进程的初始化函数。

    此函数在每个子进程启动时由 `multiprocessing.Pool` 调用一次。它的核心
    作用是为每个 worker 进程设置好一次性的、进程级的环境，使其能够正确
    地接收任务、访问共享资源并报告状态。
    [v1.5.4 修复] 放宽了对 queue 类型的检查，以兼容 Manager().Queue() 返回的代理对象。

    核心职责:
    1.  **配置日志系统**: 为每个 worker 进程配置独立的日志记录器。
    2.  **设置进程级全局变量**: 创建并初始化 `worker_info`, `progress_queue`, `global_shm_arrays`。

    Args:
        log_level (int): 主进程的日志级别。
        log_format (str): 主进程的日志格式字符串。
        queue (Optional[mp.Queue]): 用于从 worker 向主进程发送信号的共享队列或代理对象。
        shm_arrays_from_main (Dict[str, RawArray]): 初始的共享内存数组字典。
    """
    # 声明我们将要创建和修改的、进程级的全局变量
    global worker_info, progress_queue, global_shm_arrays
    
    # 获取当前 worker 进程的唯一进程ID (PID)，用于日志记录
    worker_pid = os.getpid()

    # --- 步骤 1: 严格的输入验证 ---
    if not isinstance(log_level, int):
        print(f"[Worker PID: {worker_pid}] FATAL ERROR: 'log_level' must be an integer, but got {type(log_level).__name__}.", file=sys.stderr)
        return
    if not isinstance(log_format, str):
        print(f"[Worker PID: {worker_pid}] FATAL ERROR: 'log_format' must be a string, but got {type(log_format).__name__}.", file=sys.stderr)
        return
        
    # --- [核心修复：放宽对队列类型的检查，采用鸭子类型] ---
    # 检查对象是否具有 put 和 get 方法，而不是检查其具体类型。
    # 这能同时兼容 mp.Queue 和 mp.Manager().Queue() 的 AutoProxy 对象。
    if queue is not None and not (hasattr(queue, 'put') and hasattr(queue, 'get')):
        print(f"[Worker PID: {worker_pid}] FATAL ERROR: 'queue' object does not behave like a queue (missing put/get methods). Type received: {type(queue).__name__}.", file=sys.stderr)
        return
    # --- [修复结束] ---
    
    if not isinstance(shm_arrays_from_main, dict):
        print(f"[Worker PID: {worker_pid}] FATAL ERROR: 'shm_arrays_from_main' must be a dictionary, but got {type(shm_arrays_from_main).__name__}.", file=sys.stderr)
        return
    
    # --- 步骤 2: 配置子进程的日志记录 ---
    try:
        worker_logger = logging.getLogger()
        
        if worker_logger.hasHandlers():
            worker_logger.handlers.clear()
            
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)
        worker_logger.setLevel(log_level)

    except Exception as e:
        print(f"[Worker PID: {worker_pid}] FATAL ERROR during logging configuration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return

    # --- 步骤 3: 初始化进程级全局变量 ---
    try:
        worker_info = { "pid": worker_pid }
        progress_queue = queue
        global_shm_arrays = shm_arrays_from_main

        worker_logger.debug(f"Worker process initialized successfully and is ready to receive tasks.")

    except Exception as e:
        worker_logger.critical(
            f"Worker process encountered a fatal error during global variable initialization: {e}",
            exc_info=True
        )

def enable_parallelism(num_processes: Optional[int] = None):
    """
    [公共API] [最终修正增强版] 安全地配置并启用并行计算模式。

    此函数负责设置并行计算所需的全局参数（例如，要使用的工作进程数量），
    并初始化一个可以在多次并行计算调用之间复用的共享 `multiprocessing.Manager`
    和 `Queue`。这个队列专门用于从工作进程向主进程实时报告进度和错误。

    核心设计:
    -   **临时进程池架构**: 此函数本身不创建持久化的进程池。相反，它配置
        全局参数，以便在每次需要执行大规模计算时，可以动态地、临时地创建
        专用的进程池。这确保了在 Windows 系统上共享内存的健壮性。
    -   **智能进程数选择**: 如果 `num_processes` 未指定，函数会根据系统 CPU
        核心数自动确定，但会将其限制在一个内部设定的安全上限内（默认为 8），
        以避免在内存密集型任务中耗尽系统内核资源（尤其是在 Windows 上）。
    -   **线程安全**: 使用 `_pool_lock` 锁来保护对全局状态变量的修改，
        防止在多线程环境中可能出现的竞争条件。
    -   **平台特定检查**: 在 Windows 上，会检查此函数是否在 `if __name__ == '__main__':`
        块内被调用，以防止 `multiprocessing` 产生递归的子进程错误。
    -   **幂等性**: 如果并行模式已被启用，再次调用此函数会发出警告并直接返回，
        不会重复初始化。

    此函数应该在主应用程序的开头，并在 `if __name__ == '__main__':` 块
    的保护下被调用一次。

    Args:
        num_processes (Optional[int]):
            在创建临时进程池时要启动的工作进程数量。如果为 None (默认)，
            则会自动确定一个合理的数量。

    Raises:
        ValueError: 如果用户指定的 `num_processes` 不是一个有效的正整数。
        RuntimeError: 如果在设置多进程管理器时发生无法恢复的错误。
    """
    # 声明我们将要修改的模块级全局变量
    global _parallel_enabled, _num_processes, _progress_queue
    
    # 获取日志记录器
    logger = logging.getLogger(__name__)

    # --- 步骤 1: 使用锁确保线程安全 ---
    # 防止多个线程同时尝试配置并行模式。
    with _pool_lock:
        
        # --- 步骤 2: 检查是否已经启用 (幂等性) ---
        if _parallel_enabled:
            logger.warning("Parallelism is already enabled. Call disable_parallelism() first to reconfigure.")
            return

        # --- 步骤 3: 确定要使用的工作进程数量 ---
        if num_processes is None:
            # --- 自动确定进程数 ---
            try:
                cpu_cores = os.cpu_count()
                
                # 为内存密集型的大规模模拟设置一个安全上限，以避免在 Windows 上
                # 因创建过多进程和内存映射而耗尽内核资源（如分页池），导致 OSError 1450。
                max_processes_for_large_sim = 8 
                
                if cpu_cores:
                    # 取 CPU 核心数和安全上限中的较小值
                    _num_processes = min(cpu_cores, max_processes_for_large_sim)
                    if cpu_cores > max_processes_for_large_sim:
                        logger.info(
                            f"Detected {cpu_cores} CPU cores, but limiting parallel workers to {max_processes_for_large_sim} "
                            "to conserve system resources for large-scale simulations. "
                            "This is a safeguard against OS-level resource exhaustion."
                        )
                else:
                    # 如果无法检测到 CPU 核心数，则回退到一个保守的默认值
                    _num_processes = 2
                    logger.warning("Could not determine CPU count, defaulting to 2 worker processes for parallelism.")

            except NotImplementedError:
                _num_processes = 2
                logger.warning("os.cpu_count() is not implemented on this system, defaulting to 2 worker processes.")
        else:
            # --- 使用用户指定的进程数 ---
            if not isinstance(num_processes, int) or num_processes <= 0:
                logger.error(f"Number of processes must be a positive integer, but got {num_processes}. Parallelism will NOT be enabled.")
                # 重新抛出 ValueError，因为这是来自用户的无效输入
                raise ValueError(f"Number of processes must be a positive integer, but got {num_processes}.")
            _num_processes = num_processes
        
        # --- 步骤 4: 平台特定检查 (Windows) ---
        # 在 Windows 上，`multiprocessing` 使用 'spawn' 启动方法，
        # 这要求所有与多进程相关的初始化代码都在 `if __name__ == '__main__':` 块内。
        if sys.platform.startswith('win'):
            # `mp.current_process().name` 在主进程中通常是 'MainProcess'
            if mp.current_process().name != 'MainProcess':
                logger.error(
                    "On Windows, enable_parallelism() must be called from within an `if __name__ == '__main__':` block "
                    "to prevent multiprocessing errors. Parallelism will NOT be enabled."
                )
                # 这是一个致命的配置错误，不应继续
                raise RuntimeError("On Windows, enable_parallelism() must be called from the main script execution block.")

        # --- 步骤 5: 初始化共享资源并设置全局标志 ---
        try:
            # 创建一个可以在多次并行调用之间复用的 Manager 和 Queue。
            # Manager 会启动一个后台服务进程来管理共享对象，复用它可以避免
            # 为每次并行计算都重复创建这个服务进程。
            manager = mp.Manager()
            _progress_queue = manager.Queue()
            
            # 设置全局标志位，表示并行模式的配置已完成，可以按需启动并行任务。
            _parallel_enabled = True
            
            logger.info(f"Parallelism has been CONFIGURED and ENABLED. On-demand pools will use up to {_num_processes} worker processes.")

        except Exception as e:
            # 捕获在 Manager 初始化过程中可能发生的任何错误
            logger.critical(f"Failed to set up multiprocessing manager and enable parallelism: {e}", exc_info=True)
            # 确保在失败时状态被安全地重置
            _parallel_enabled = False
            _num_processes = 0
            _progress_queue = None
            # 重新抛出异常，让调用者知道配置失败
            raise RuntimeError("Failed to initialize the multiprocessing manager.") from e
def disable_parallelism():
    """
    [公共API] [最终修正增强版] 安全地禁用并行计算模式并清理相关共享资源。

    此函数负责将量子核心库从并行模式切换回串行模式。在当前“临时进程池”
    的架构下，此函数不再负责关闭一个持久化的进程池，因为进程池是为每个
    计算任务动态创建和销__销毁的。

    它的核心职责是：
    1.  **线程安全地**将全局并行开关 `_parallel_enabled` 设为 `False`。
    2.  重置全局的进程数配置 `_num_processes` 为 0。
    3.  清理在 `enable_parallelism` 中创建的可复用共享资源，主要是
        `multiprocessing.Manager` 所管理的后台服务进程和共享队列。通过将
        全局变量 `_progress_queue` 的引用设为 `None`，可以帮助 Python 的
        垃圾回收机制回收 `Manager` 对象，从而优雅地关闭其后台进程。

    在应用程序退出前或需要切换回串行模式时调用此函数是一个良好的实践，
    以确保所有与多进程相关的后台服务都能被干净地关闭。

    核心增强功能:
    -   **线程安全**: 使用 `_pool_lock` 锁来保护对全局状态变量
        (`_parallel_enabled`, `_num_processes`, `_progress_queue`) 的修改，
        防止在多线程环境中可能出现的竞争条件。
    -   **幂等性 (Idempotency)**: 如果并行模式尚未启用，函数会简单地记录
        一条信息并直接返回，多次调用不会产生副作用。
    -   **清晰的日志记录**: 提供了明确的日志信息，告知用户并行模式的状态变化。
    """
    # 声明我们将要修改的模块级全局变量
    global _parallel_enabled, _num_processes, _progress_queue
    
    # 获取日志记录器
    logger = logging.getLogger(__name__)

    # --- 步骤 1: 使用锁确保线程安全 ---
    # 在多线程应用中，多个线程可能同时尝试禁用并行模式。
    # 使用锁可以确保同一时间只有一个线程能够修改这些全局状态变量。
    with _pool_lock:
        
        # --- 步骤 2: 检查并行模式是否已启用 (幂等性) ---
        if not _parallel_enabled:
            logger.info("Parallelism was already disabled. No action taken.")
            return
            
        # --- 步骤 3: 重置所有与并行相关的全局状态变量 ---
        
        logger.debug("Disabling parallelism and preparing to release shared resources...")

        # 这是核心的禁用逻辑：将全局开关关闭，并清除进程数配置。
        _parallel_enabled = False
        _num_processes = 0
        
        # --- 步骤 4: 清理共享资源 ---
        
        # 将共享队列的引用设置为 None。
        # 创建这个队列的 `mp.Manager` 对象是在 `enable_parallelism` 函数中
        # 创建的。当该函数结束时，如果没有其他地方（比如 _progress_queue）
        # 引用这个 Manager 创建的对象，Python 的垃圾回收机制就会处理它，
        # 从而关闭其后台的管理服务进程。将 _progress_queue 设为 None
        # 是向垃圾回收器发出的一个明确信号，表明该对象可以被回收了。
        if _progress_queue is not None:
            # 尽管我们不能显式地关闭 Manager，但清除引用是最佳实践。
            _progress_queue = None
            logger.debug("Shared progress queue reference has been cleared for garbage collection.")
        
        # --- 步骤 5: 记录操作完成 ---
        logger.info("Parallelism has been successfully DISABLED. The system will now operate in serial mode.")


# --- 内部并行工作函数 ---
# 这些函数必须在模块的顶层定义，以便它们可以被子进程序列化和访问。

def _pure_python_dot_product(mat_a: List[List[float]], mat_b: List[List[float]]) -> List[List[float]]:
    """
    [最终修正增强版] 一个独立的、纯Python实现的、仅处理浮点数矩阵的乘法函数 (C = A @ B)。

    此版本经过全面加固，包含了对输入类型、形状和数值内容的详尽验证，
    以确保在多进程的 worker 进程中独立运行时的高度健壮性。

    核心增强功能:
    -   **严格的类型检查**: 验证输入 `mat_a` 和 `mat_b` 必须是嵌套列表。
    -   **形状与维度验证**:
        -   确保矩阵非空。
        -   确保所有行具有相同的长度（形状规整）。
        -   验证内积维度是否匹配 (`cols_a == rows_b`)。
    -   **数值类型检查**: (可选，但推荐) 验证嵌套列表中的所有元素都是数值类型
        （`float` 或 `int`），防止因非数值内容导致的 `TypeError`。
    -   **清晰的错误信息**: 在验证失败时，提供详细的错误信息，指明哪个矩阵、
        哪一行或哪个维度出了问题，极大地方便了调试。

    Args:
        mat_a (List[List[float]]):
            乘法中的左矩阵 (A)。必须是一个非空的、形状规整的浮点数嵌套列表。
        mat_b (List[List[float]]):
            乘法中的右矩阵 (B)。必须是一个非空的、形状规整的浮点数嵌套列表。

    Returns:
        List[List[float]]:
            结果矩阵 C，其形状为 (rows_a, cols_b)。

    Raises:
        TypeError: 如果输入 `mat_a` 或 `mat_b` 不是嵌套列表或包含非数值元素。
        ValueError: 如果任一矩阵为空、形状不规整（各行长度不一），
                    或者它们的维度不满足矩阵乘法的要求。
    """
    # --- 步骤 1: 严格的输入验证 ---

    # -- 验证矩阵 A --
    if not isinstance(mat_a, list) or not mat_a or not isinstance(mat_a[0], list):
        raise TypeError("Input 'mat_a' must be a non-empty nested list (matrix).")
    
    rows_a = len(mat_a)
    cols_a = len(mat_a[0])

    if cols_a == 0:
        raise ValueError("Input 'mat_a' cannot have rows of zero length.")

    for i in range(rows_a):
        row = mat_a[i]
        if not isinstance(row, list) or len(row) != cols_a:
            raise ValueError(
                f"Input 'mat_a' is not a well-formed matrix. Row {i} has length {len(row)}, but expected {cols_a}."
            )
        # (可选但推荐) 检查行内元素类型
        for val in row:
            if not isinstance(val, (float, int)):
                raise TypeError(f"Input 'mat_a' contains a non-numeric element '{val}' (type: {type(val).__name__}) at row {i}.")

    # -- 验证矩阵 B --
    if not isinstance(mat_b, list) or not mat_b or not isinstance(mat_b[0], list):
        raise TypeError("Input 'mat_b' must be a non-empty nested list (matrix).")

    rows_b = len(mat_b)
    cols_b = len(mat_b[0])

    if cols_b == 0:
        raise ValueError("Input 'mat_b' cannot have rows of zero length.")

    for i in range(rows_b):
        row = mat_b[i]
        if not isinstance(row, list) or len(row) != cols_b:
            raise ValueError(
                f"Input 'mat_b' is not a well-formed matrix. Row {i} has length {len(row)}, but expected {cols_b}."
            )
        # (可选但推荐) 检查行内元素类型
        for val in row:
            if not isinstance(val, (float, int)):
                raise TypeError(f"Input 'mat_b' contains a non-numeric element '{val}' (type: {type(val).__name__}) at row {i}.")
    
    # -- 验证维度兼容性 --
    # A 的列数必须等于 B 的行数
    if cols_a != rows_b:
        raise ValueError(
            f"Matrix dimensions are incompatible for dot product: A is ({rows_a}x{cols_a}) "
            f"and B is ({rows_b}x{cols_b}). The inner dimensions ({cols_a} and {rows_b}) must match."
        )

    # --- 步骤 2: 核心计算 ---

    # 使用列表推导式高效地创建一个正确大小的全零结果矩阵
    res = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    # 核心的三层嵌套循环，用于实现 C[i,j] = Σ_k (A[i,k] * B[k,j])
    for i in range(rows_a):      # 遍历结果矩阵的行
        for j in range(cols_b):  # 遍历结果矩阵的列
            # 为了性能，可以预先提取行和列
            row_a_i = mat_a[i]
            # col_b_j = [mat_b[k][j] for k in range(rows_b)] # 创建列向量会增加开销
            
            sum_val = 0.0
            for k in range(cols_a):  # 遍历内积维度 (A的列或B的行)
                sum_val += row_a_i[k] * mat_b[k][j]
            
            res[i][j] = sum_val
            
    # --- 步骤 3: 返回结果 ---
    return res
def _worker_apply_unitary_chunk_pure(task_info: Dict[str, Any]) -> bool:
    """
    [最终修正增强版] 并行酉变换的工作进程目标函数 (ρ' = UρU†)。
    [v1.5.2 刷屏问题修正版] 增强了异常报告机制。
    """
    global worker_info, progress_queue, global_shm_arrays
    import traceback

    pid = os.getpid()
    local_progress_queue = None

    try:
        if 'worker_info' not in globals() or 'progress_queue' not in globals() or 'global_shm_arrays' not in globals():
            raise RuntimeError("Worker process was not initialized correctly.")
        
        local_progress_queue = progress_queue

        if global_shm_arrays is None:
            raise RuntimeError("Shared memory arrays reference is None.")

        row_chunk = task_info.get('row_chunk')
        dim = task_info.get('dim')
        if row_chunk is None or dim is None:
            raise ValueError(f"Task info dictionary is incomplete: {task_info}")
        
        start_row, end_row = row_chunk
        chunk_size = end_row - start_row
        
        if chunk_size <= 0:
            if local_progress_queue:
                local_progress_queue.put(1)
            return True

        required_keys = {'U_real', 'U_imag', 'rho_real', 'rho_imag', 'result_real', 'result_imag'}
        if not required_keys.issubset(global_shm_arrays.keys()):
            raise KeyError(f"Shared memory is missing required keys. Expected {required_keys}, got {global_shm_arrays.keys()}")

        U_real = [global_shm_arrays['U_real'][i*dim:(i+1)*dim] for i in range(dim)]
        U_imag = [global_shm_arrays['U_imag'][i*dim:(i+1)*dim] for i in range(dim)]
        rho_real = [global_shm_arrays['rho_real'][i*dim:(i+1)*dim] for i in range(dim)]
        rho_imag = [global_shm_arrays['rho_imag'][i*dim:(i+1)*dim] for i in range(dim)]
        
        U_chunk_real = U_real[start_row:end_row]
        U_chunk_imag = U_imag[start_row:end_row]

        temp_AC = _pure_python_dot_product(U_chunk_real, rho_real)
        temp_BD = _pure_python_dot_product(U_chunk_imag, rho_imag)
        temp_real = [[ac - bd for ac, bd in zip(row_ac, row_bd)] for row_ac, row_bd in zip(temp_AC, temp_BD)]

        temp_AD = _pure_python_dot_product(U_chunk_real, rho_imag)
        temp_BC = _pure_python_dot_product(U_chunk_imag, rho_real)
        temp_imag = [[ad + bc for ad, bc in zip(row_ad, row_bc)] for row_ad, row_bc in zip(temp_AD, temp_BC)]

        U_T_real = [[U_real[j][i] for j in range(dim)] for i in range(dim)]
        U_T_imag = [[U_imag[j][i] for j in range(dim)] for i in range(dim)]
        
        res_AC_final = _pure_python_dot_product(temp_real, U_T_real)
        res_BD_final = _pure_python_dot_product(temp_imag, U_T_imag)
        result_chunk_real = [[ac + bd for ac, bd in zip(row_ac, row_bd)] for row_ac, row_bd in zip(res_AC_final, res_BD_final)]

        res_BC_final = _pure_python_dot_product(temp_imag, U_T_real)
        res_AD_final = _pure_python_dot_product(temp_real, U_T_imag)
        result_chunk_imag = [[bc - ad for bc, ad in zip(row_bc, row_ad)] for row_bc, row_ad in zip(res_BC_final, res_AD_final)]
        
        for i in range(chunk_size):
            global_row_idx = start_row + i
            start_pos = global_row_idx * dim
            end_pos = start_pos + dim
            global_shm_arrays['result_real'][start_pos:end_pos] = result_chunk_real[i]
            global_shm_arrays['result_imag'][start_pos:end_pos] = result_chunk_imag[i]

        if local_progress_queue:
            local_progress_queue.put(1)
        return True

    except Exception as e:
        # [核心修改]
        worker_logger = logging.getLogger(__name__)
        worker_logger.error(
            f"Worker process (PID: {pid}) failed on unitary evolution chunk {task_info.get('row_chunk', 'N/A')}: {e}",
            exc_info=True
        )
        
        error_info = {
            "type": "error",
            "pid": pid,
            "chunk": task_info.get('row_chunk', 'N/A'),
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc()
        }
        
        if local_progress_queue:
            try:
                local_progress_queue.put(error_info)
            except Exception as q_e:
                worker_logger.critical(f"Failed to send error report to main process via queue: {q_e}")

        return False
def _worker_set_shm(shm_arrays_for_task: Dict[str, RawArray]):
    """
    [最终修正增强版] 一个在 worker 进程中执行的简单任务，用于安全地更新
    该进程对全局共享内存（`global_shm_arrays`）的引用。

    在 Windows 系统上，由于进程是通过 `spawn` 创建的，子进程不会继承父进程
    的内存空间。因此，共享内存对象（如 `RawArray`）必须通过进程池的
    `initializer` 或显式地作为参数传递给任务。

    此函数用于一种**动态更新共享内存**的模式：当主进程为一项新的并行计算
    任务（例如，一次大的矩阵乘法）创建了一组新的共享内存数组后，它会首先
    向进程池中的所有 worker 广播这个 `_worker_set_shm` 任务。每个 worker
    执行此任务时，就会将其进程级的全局变量 `global_shm_arrays` 指向这组
    新的共享内存，为后续真正的计算任务做好准备。

    核心增强功能:
    -   **健壮的初始化检查**: 验证进程级全局变量 `global_shm_arrays` 是否存在，
        确保 `init_worker` 已被正确调用。
    -   **严格的输入验证**: 检查传入的 `shm_arrays_for_task` 是否为预期的
        字典类型，并且其值是 `RawArray` 对象。
    -   **清晰的日志记录**: 在 `DEBUG` 级别下记录共享内存的更新操作，
        包括 worker 的 PID 和新的共享内存对象的 ID，便于追踪内存管理。

    Args:
        shm_arrays_for_task (Dict[str, RawArray]):
            一个字典，其键是共享内存的名称（如 'U_real'），值是主进程创建的
            `multiprocessing.RawArray` 对象。

    Returns:
        bool: 始终返回 `True` 表示成功，或在发生严重错误时由异常中断。
              `apply_async` 的结果将捕获成功或失败状态。
    """
    # 声明我们将要修改的、在 `init_worker` 中创建的进程级全局变量
    global worker_info, global_shm_arrays
    
    # 获取当前 worker 的 PID，用于日志记录
    pid = os.getpid()
    worker_logger = logging.getLogger(__name__)

    try:
        # --- 步骤 1: 严格的验证 ---
        
        # 验证 worker 是否已被正确初始化
        if 'global_shm_arrays' not in globals():
            # 这是一个致命错误，表明 `init_worker` 未能按预期运行
            raise RuntimeError("Worker process is missing the 'global_shm_arrays' global variable. Initialization failed.")

        # 验证传入的参数类型
        if not isinstance(shm_arrays_for_task, dict):
            raise TypeError(f"Expected a dictionary for 'shm_arrays_for_task', but received {type(shm_arrays_for_task).__name__}.")

        # (可选但推荐) 验证字典内容的基本类型
        for key, value in shm_arrays_for_task.items():
            if not isinstance(value, RawArray):
                worker_logger.warning(f"Item '{key}' in 'shm_arrays_for_task' is not a RawArray instance (got {type(value).__name__}). This might be unexpected.")

        # --- 步骤 2: 更新全局引用 ---

        worker_logger.debug(
            f"Worker (PID: {pid}) is updating its reference to shared memory arrays. "
            f"Old ID: {id(global_shm_arrays) if global_shm_arrays is not None else 'None'}, "
            f"New ID: {id(shm_arrays_for_task)}."
        )
        
        global_shm_arrays = shm_arrays_for_task
        
        # --- 步骤 3: 返回成功状态 ---
        
        return True

    except Exception as e:
        # --- 步骤 4: 捕获并记录任何意外错误 ---
        worker_logger.critical(
            f"Worker (PID: {pid}) encountered a fatal error during _worker_set_shm: {e}",
            exc_info=True
        )
        # 在多进程任务中，最好让异常自然冒泡，
        # 主进程的 `result.get()` 会捕获并处理它。
        raise

def _worker_apply_h_chunk(task_info: Dict[str, Any]) -> bool:
    """
    [最终修正增强版] 并行 Hadamard 变换的工作进程目标函数 (纯Python版)。
    [v1.5.14 核心逻辑修复] 实现了从“有效索引”到“态矢量索引”的正确映射，确保并行计算的正确性。
    
    此函数在每个并行的 worker 进程中执行，负责计算 Hadamard 变换的一部分。
    它从共享内存中读取输入态矢量的一个片段，执行计算，然后将结果写回到
    输出共享内存中。

    核心修正:
    - 索引映射: 主进程将整个 `dim`-维的态矢量划分为 `dim/2` 个“对”，
      并分发这些“对”的索引 `k` (范围从 0 到 dim/2 - 1) 给 worker。
      此函数的核心任务是将这个“对索引” `k` 正确地映射回态矢量中
      需要操作的两个实际索引 `i` 和 `j`。
      - `i` 是满足 `(i >> target_qubit) & 1 == 0` 的索引。
      - `j` 是满足 `j = i | (1 << target_qubit)` 的配对索引。
    这个修正确保了变换作用在正确的振幅对上。

    Args:
        task_info (Dict[str, Any]):
            一个字典，包含了此工作块所需的所有信息：
            - 'chunk_range' (Tuple[int, int]): 此 worker 负责处理的“有效索引” `k` 的范围 `(start_idx, end_idx)`。
            - 'dim' (int): 整个态矢量的维度 (2^N)。
            - 'target_qubit' (int): Hadamard 门作用的目标量子比特。

    Returns:
        bool: 如果计算成功，返回 `True`；如果发生错误，返回 `False`。
              返回 `False` 也会通过共享队列向主进程报告详细错误。
    """
    # --- 步骤 1: 导入必要的模块并设置进程级全局变量 ---
    import traceback
    # 声明我们将要访问的、在 init_worker 中创建的进程级全局变量
    global worker_info, progress_queue, global_shm_arrays
    
    # 获取当前 worker 的 PID 和共享队列的引用，用于日志和状态报告
    pid = os.getpid()
    local_progress_queue = None
    
    try:
        # --- 步骤 2: 严格的初始化和参数验证 ---
        
        # a) 检查 worker 进程是否已被正确初始化
        if 'worker_info' not in globals() or 'progress_queue' not in globals() or 'global_shm_arrays' not in globals():
            raise RuntimeError("Worker process was not initialized correctly via init_worker.")
        
        local_progress_queue = progress_queue

        if global_shm_arrays is None:
            raise RuntimeError("Shared memory arrays reference (global_shm_arrays) is None in worker process.")

        # b) 从任务字典中安全地提取参数
        chunk_range = task_info.get('chunk_range')
        dim = task_info.get('dim')
        target_qubit = task_info.get('target_qubit')

        if chunk_range is None or dim is None or target_qubit is None:
            raise ValueError(f"Task info dictionary is incomplete: {task_info}")
        
        start_idx, end_idx = chunk_range

        # c) 检查共享内存中是否存在必需的数组
        required_shm_keys = {'sv_in', 'sv_out'}
        if not required_shm_keys.issubset(global_shm_arrays.keys()):
            raise KeyError(f"Shared memory is missing required keys. Expected {required_shm_keys}, got {global_shm_arrays.keys()}")

        # d) 获取对共享内存数组的引用
        sv_in = global_shm_arrays['sv_in']
        sv_out = global_shm_arrays['sv_out']

        # --- 步骤 3: 准备计算所需的常量 ---
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        target_mask = 1 << target_qubit

        # --- 步骤 4: [核心修正] 执行计算循环 ---
        # 循环变量 `k` 是主进程分发的“有效索引”，范围是 [0, dim/2 - 1]。
        # 我们必须将 `k` 映射回正确的态矢量索引 `i`，其中 `i` 的 target_qubit 位为 0。
        for k in range(start_idx, end_idx):
            
            # a) 将块索引 k 映射回正确的态矢量索引 i (其中 target_qubit 位为 0)
            # 这个位运算逻辑是修正的核心，它通过在 target_qubit 的位置“插入”一个0来重构索引。
            lower_bits = k & (target_mask - 1)
            higher_bits = k >> target_qubit
            i = (higher_bits << (target_qubit + 1)) | lower_bits
            
            # b) `i` 现在是正确的态矢量索引，其 target_qubit 位为 0。
            #    `j` 是与之配对的、target_qubit 位为 1 的索引。
            j = i | target_mask
            
            # c) 从共享内存中读取一对振幅。每个复数由两个连续的 double (实部, 虚部) 表示。
            base_idx_i = 2 * i
            base_idx_j = 2 * j
            
            amp0_real, amp0_imag = sv_in[base_idx_i], sv_in[base_idx_i + 1]
            amp1_real, amp1_imag = sv_in[base_idx_j], sv_in[base_idx_j + 1]
            
            # d) 根据 Hadamard 变换的定义计算新的振幅
            # new_amp0 = (amp0 + amp1) / sqrt(2)
            # new_amp1 = (amp0 - amp1) / sqrt(2)
            new_amp0_real = (amp0_real + amp1_real) * sqrt2_inv
            new_amp0_imag = (amp0_imag + amp1_imag) * sqrt2_inv
            
            new_amp1_real = (amp0_real - amp1_real) * sqrt2_inv
            new_amp1_imag = (amp0_imag - amp1_imag) * sqrt2_inv
            
            # e) 将计算结果写回到输出共享内存中
            sv_out[base_idx_i] = new_amp0_real
            sv_out[base_idx_i + 1] = new_amp0_imag
            sv_out[base_idx_j] = new_amp1_real
            sv_out[base_idx_j + 1] = new_amp1_imag
        # --- [核心修正结束] ---

        # --- 步骤 5: 向主进程报告任务完成 ---
        if local_progress_queue:
            local_progress_queue.put(1) # '1' 表示成功完成一个任务块
            
        return True

    except Exception as e:
        # --- 步骤 6: 健壮的错误处理 ---
        # 如果在 worker 进程中发生任何未捕获的异常，记录它并通过共享队列向主进程报告。
        worker_logger = logging.getLogger(__name__)
        worker_logger.error(
            f"Worker process (PID: {pid}) failed on Hadamard chunk {task_info.get('chunk_range', 'N/A')}: {e}",
            exc_info=True
        )
        
        # 构建一个包含详细错误信息的字典
        error_info = {
            "type": "error",
            "pid": pid,
            "chunk": task_info.get('chunk_range', 'N/A'),
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc()
        }
        
        # 尝试将错误报告发送回主进程
        if local_progress_queue:
            try:
                local_progress_queue.put(error_info)
            except Exception as q_e:
                # 如果连队列通信都失败了，只能在本地日志中记录这个严重问题
                worker_logger.critical(f"Failed to send error report to main process via queue: {q_e}")

        return False

def _execute_parallel_unitary_evolution_pure(unitary: List[List[complex]], rho: List[List[complex]]) -> Optional[List[List[complex]]]:
    """
    [v1.5.2 刷屏问题修正版] 在主进程中编排纯Python版的并行酉变换 (ρ' = UρU†)。
    此版本修正了当多个 worker 同时失败时，错误信息会重复打印（刷屏）的问题。
    """
    global _parallel_enabled, _num_processes, _progress_queue

    if not _parallel_enabled or _num_processes <= 0:
        logger.error("Internal parallel executor called, but parallelism is not enabled. Call enable_parallelism() first.")
        return None

    if not rho: return []
    dim = len(rho)
    if dim == 0: return []
    size = dim * dim
    
    total_shm_bytes = 6 * size * ctypes.sizeof(ctypes.c_double)
    total_shm_gb = total_shm_bytes / _GB_TO_BYTES
    logger.info(f"Allocating approximately {total_shm_gb:.3f} GB of shared memory for parallel density matrix evolution.")
    
    try:
        shm_arrays = {
            'U_real': RawArray(ctypes.c_double, size), 'U_imag': RawArray(ctypes.c_double, size),
            'rho_real': RawArray(ctypes.c_double, size), 'rho_imag': RawArray(ctypes.c_double, size),
            'result_real': RawArray(ctypes.c_double, size), 'result_imag': RawArray(ctypes.c_double, size),
        }
    except Exception as e:
        logger.critical(f"Failed to allocate {total_shm_gb:.2f} GB of shared memory for this task: {e}", exc_info=True)
        return None

    def fill_shm(matrix: List[List[complex]], real_arr: RawArray, imag_arr: RawArray):
        real_arr[:] = [elem.real for row in matrix for elem in row]
        imag_arr[:] = [elem.imag for row in matrix for elem in row]

    try:
        fill_shm(unitary, shm_arrays['U_real'], shm_arrays['U_imag'])
        fill_shm(rho, shm_arrays['rho_real'], shm_arrays['rho_imag'])
    except Exception as e:
        logger.critical(f"Failed to fill shared memory with matrix data for this task: {e}", exc_info=True)
        return None

    num_chunks = _num_processes * 4
    chunk_size = max(1, math.ceil(dim / num_chunks)) 
    row_chunks = [(i, min(i + chunk_size, dim)) for i in range(0, dim, chunk_size)]
    tasks = [{'row_chunk': chunk, 'dim': dim} for chunk in row_chunks]
    
    pool_for_this_run: Optional[pool.Pool] = None
    all_workers_succeeded = False
    fatal_error_from_worker = None

    try:
        root_logger = logging.getLogger()
        log_level = root_logger.level if root_logger.level != 0 else logging.INFO
        log_format = "%(asctime)s - [%(levelname)s] - (%(name)s) - [PID:%(process)d] - %(message)s"
        init_args = (log_level, log_format, _progress_queue, shm_arrays)
        
        ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
        pool_for_this_run = ctx.Pool(processes=_num_processes, initializer=init_worker, initargs=init_args)

        total_tasks = len(tasks)
        results = [pool_for_this_run.apply_async(_worker_apply_unitary_chunk_pure, args=(task,)) for task in tasks]
        logger.info(f"Asynchronously submitted {total_tasks} micro-chunks to a temporary pool of {_num_processes} workers.")

        completed_tasks = 0
        print_progress_bar(0, total_tasks, prefix='Progress:', suffix='Complete', length=50)
        
        # [核心修正] 带有快速失败机制的监控循环
        while completed_tasks < total_tasks:
            try:
                message = _progress_queue.get(timeout=0.2)
                
                if isinstance(message, dict) and message.get("type") == "error":
                    if fatal_error_from_worker is None:
                        fatal_error_from_worker = message
                    break
                
                elif message == 1:
                    completed_tasks += 1
                    print_progress_bar(completed_tasks, total_tasks, prefix='Progress:', suffix='Complete', length=50)
                else:
                    logger.warning(f"Received unknown message from progress queue: {message}")

            except mp.queues.Empty:
                if any(res.ready() and not res.successful() for res in results):
                    logger.error("A worker process has terminated unexpectedly (silent failure). Aborting.")
                    break 
                time.sleep(0.05)
        
        sys.stdout.write('\n')
        sys.stdout.flush()

        # [核心修正] 循环结束后的统一处理
        if fatal_error_from_worker:
            raise RuntimeError("A worker process reported a fatal error.")
        
        if completed_tasks < total_tasks:
            for res in results:
                if res.ready() and not res.successful():
                    try:
                        res.get()
                    except Exception as worker_exc:
                        logger.error(f"Caught exception from a silently failed worker: {worker_exc}", exc_info=True)
                        break
            raise RuntimeError("One or more worker processes failed during execution.")

        final_results = [res.get() for res in results]
        all_workers_succeeded = all(final_results)
            
    except (KeyboardInterrupt, SystemExit):
        sys.stdout.write('\n')
        logger.warning("Parallel execution was manually interrupted by the user.")
        all_workers_succeeded = False
    except Exception as e:
        sys.stdout.write('\n')
        logger.critical(f"A critical error was raised during parallel execution orchestration: {e}", exc_info=False)
        if fatal_error_from_worker:
            logger.error(f"--- Full Traceback from Failing Worker (PID: {fatal_error_from_worker.get('pid')}) ---\n"
                                        f"{fatal_error_from_worker.get('traceback', 'No traceback available.')}")
        all_workers_succeeded = False
    finally:
        if pool_for_this_run:
            logger.debug("Terminating the temporary process pool for this computation...")
            if not all_workers_succeeded:
                pool_for_this_run.terminate()
            pool_for_this_run.close()
            pool_for_this_run.join()
            logger.debug("Temporary process pool terminated.")
        
        if not all_workers_succeeded:
            return None

    final_rho_real_flat = list(shm_arrays['result_real'])
    final_rho_imag_flat = list(shm_arrays['result_imag'])
    
    final_rho = [
        [complex(final_rho_real_flat[i*dim + j], final_rho_imag_flat[i*dim + j]) for j in range(dim)]
        for i in range(dim)
    ]
    
    return final_rho
# ========================================================================
# --- [新架构] 内部辅助类：用于实际计算的态矢量实体 ---
# ========================================================================

class _StateVectorEntity:
    """
    [健壮性改进版] 一个内部辅助类，代表一个“实体化”的量子态，基于态矢量。
    它不具备惰性求值能力，而是直接存储和操作一个庞大的态矢量数组。
    这个类的功能类似于旧版本的 QuantumState，但仅限于态矢量，
    并且被设计为由惰性的 QuantumState 在内部使用。
    """
    
    # [内部常量] Pauli矩阵的纯Python列表表示 (作为 ClassVar 以避免 mutable default 错误)
    _SIGMA_X_LIST: ClassVar[List[List[complex]]] = [[0+0j, 1+0j], [1+0j, 0+0j]]
    _SIGMA_Y_LIST: ClassVar[List[List[complex]]] = [[0+0j, 0-1j], [0+1j, 0+0j]]
    _SIGMA_Z_LIST: ClassVar[List[List[complex]]] = [[1+0j, 0+0j], [0+0j, -1+0j]]
    _IDENTITY_LIST: ClassVar[List[List[complex]]] = [[1+0j, 0+0j], [0+0j, 1+0j]]

    def __init__(self, num_qubits: int, backend: Any):
        """
        [最终修正增强版] 初始化一个态矢量实体。

        此版本对内存占用预估和检查逻辑进行了全面加固，以防止因分配过大
        内存（RAM 或 VRAM）而导致的程序或系统崩溃。

        核心增强功能:
        -   **分层内存限制检查**: 内存检查逻辑会按以下优先级确定可用内存上限：
            1.  **用户配置**: 优先使用通过 `configure_quantum_core` 设置的
                `MAX_SYSTEM_RAM_GB_LIMIT`。
            2.  **动态检测**: 如果用户未配置，则尝试通过 `_get_available_system_ram_gb`
                动态检测当前系统的可用物理内存。
            3.  **硬编码回退**: 如果以上方法均失败，则使用一个保守的硬编码值
                `_TOTAL_SYSTEM_RAM_GB_HW` 作为最后的防线。
        -   **清晰的日志反馈**: 在进行内存检查时，会明确记录所使用的内存上限的
            来源（用户配置、动态检测或硬编码值），便于调试。
        -   **安全的内存使用系数**: 在与可用内存比较时，会乘以一个安全系数
            （例如 0.85），为操作系统和其他程序保留一定的内存余量。
        -   **严格的输入验证**: 对 `num_qubits` 和 `backend` 的类型和值进行
            严格检查，确保对象在创建时处于有效状态。
        -   **完整的初始化**: 正确初始化所有实例属性，包括态矢量、后端引用、
            日志器和优化内核调度字典。

        Args:
            num_qubits (int): 量子比特数。
            backend (Any): 计算后端实例 (PurePythonBackend or CuPyBackendWrapper)。

        Raises:
            ValueError: 如果 `num_qubits` 为负数。
            TypeError: 如果 `backend` 为 None 或类型不正确。
            MemoryError: 如果预估的内存需求超过了可用的安全上限。
        """
        # --- 步骤 1: 初始化日志器并进行严格的输入验证 ---
        
        self._internal_logger = logging.getLogger(f"{_StateVectorEntity.__module__}.{_StateVectorEntity.__name__}")

        if not isinstance(num_qubits, int) or num_qubits < 0:
            self._internal_logger.error(f"_StateVectorEntity: 'num_qubits' must be a non-negative integer, got {num_qubits}.")
            raise ValueError("num_qubits must be a non-negative integer.")
        if backend is None:
            self._internal_logger.error("_StateVectorEntity: 'backend' instance cannot be None.")
            raise TypeError("A valid backend instance must be provided.")
        if not isinstance(backend, (PurePythonBackend, CuPyBackendWrapper)):
            self._internal_logger.error(f"_StateVectorEntity: 'backend' must be PurePythonBackend or CuPyBackendWrapper, got {type(backend).__name__}.")
            raise TypeError(f"Backend must be PurePythonBackend or CuPyBackendWrapper instance, but got {type(backend).__name__}.")
        
        self.num_qubits = num_qubits
        self._backend = backend
        
        dim = 1 << num_qubits

        # --- 步骤 2: 内存需求预估与分层检查 ---
        
        # 每个复数元素（complex128）占用 16 字节
        required_bytes = dim * _BYTES_PER_COMPLEX128_ELEMENT
        required_gb = required_bytes / _GB_TO_BYTES

        if isinstance(self._backend, CuPyBackendWrapper):
            # 对于 GPU，我们目前依赖硬编码的 VRAM 值，并使用一个安全系数
            available_vram_gb = _TOTAL_GPU_VRAM_GB_HW
            safe_limit_gb = available_vram_gb * 0.85 # 为其他应用和系统保留 15% 的 VRAM
            
            if required_gb > safe_limit_gb:
                error_msg = (
                    f"Memory allocation pre-check failed for {num_qubits}-qubit state on GPU. "
                    f"Required VRAM: {required_gb:.2f} GB. "
                    f"Available VRAM (estimated safe limit from hardcoded value of {available_vram_gb} GB): ~{safe_limit_gb:.2f} GB. "
                    "The number of qubits exceeds the GPU's memory capacity. "
                    "Please reduce the number of qubits or switch to a Pure Python backend on a machine with more RAM."
                )
                self._internal_logger.critical(error_msg)
                raise MemoryError(error_msg)
        else: # PurePythonBackend
            available_ram_gb = None
            limit_source = "unknown"

            # 优先级 1: 检查用户是否在全局配置中手动设置了内存上限
            user_ram_limit = _core_config.get("MAX_SYSTEM_RAM_GB_LIMIT")
            if user_ram_limit is not None and isinstance(user_ram_limit, (int, float)) and user_ram_limit > 0:
                available_ram_gb = float(user_ram_limit)
                limit_source = "user-configured"
                self._internal_logger.info(f"Using user-configured RAM limit of {available_ram_gb:.2f} GB for pre-allocation check.")
            else:
                # 优先级 2: 尝试动态检测系统可用内存
                dynamic_ram_gb = _get_available_system_ram_gb()
                if dynamic_ram_gb is not None:
                    available_ram_gb = dynamic_ram_gb
                    limit_source = "dynamically detected"
                    self._internal_logger.debug(f"Dynamically detected available system RAM: {available_ram_gb:.2f} GB.")
            
            # 优先级 3: 如果以上方法都失败，则回退到硬编码的默认值
            if available_ram_gb is None:
                available_ram_gb = _TOTAL_SYSTEM_RAM_GB_HW
                limit_source = "hardcoded fallback"
                self._internal_logger.warning(
                    f"Could not determine available system RAM dynamically or from user configuration. "
                    f"Falling back to a conservative hardcoded value of {available_ram_gb} GB."
                )

            # 使用安全系数为操作系统和其他程序保留内存
            safe_limit_gb = available_ram_gb * 0.85
            if required_gb > safe_limit_gb:
                error_msg = (
                    f"Memory allocation pre-check failed for {num_qubits}-qubit state on CPU. "
                    f"Required RAM: {required_gb:.2f} GB. "
                    f"Available RAM ({limit_source}, safe limit): ~{safe_limit_gb:.2f} GB. "
                    "The number of qubits exceeds the system's available memory. Please reduce the number of qubits."
                )
                self._internal_logger.critical(error_msg)
                raise MemoryError(error_msg)
        
        # --- 步骤 3: 尝试分配内存并初始化 |0...0> 态 ---
        try:
            self._state_vector = self._backend.zeros((dim,), dtype=complex)
        except Exception as e:
            self._internal_logger.critical(f"Failed to allocate memory for {num_qubits}-qubit state vector even after pre-check passed. Error: {e}", exc_info=True)
            raise MemoryError(f"Failed to allocate memory for {num_qubits}-qubit state vector.") from e

        # 初始化 |0...0> 态，即第一个振幅为 1
        if dim > 0:
            if isinstance(self._state_vector, list):
                self._state_vector[0] = 1.0 + 0.0j
            else: # CuPy array
                self._state_vector[0] = 1.0 + 0.0j

        # --- 步骤 4: 初始化优化内核的调度字典 ---
        # 键: 门名称; 值: (执行内核函数, 期望的位置参数数量)
        self._kernel_map: Dict[str, Tuple[Callable[..., None], int]] = {
            'x': (self._apply_x_kernel_sv, 1), 
            'y': (self._apply_y_kernel_sv, 1),
            'z': (self._apply_z_kernel_sv, 1), 
            'h': (self._apply_h_kernel_sv, 1),
            's': (self._apply_s_kernel_sv, 1),
            'sx': (self._apply_sx_kernel_sv, 1),
            'sdg': (self._apply_sdg_kernel_sv, 1),
            't_gate': (self._apply_t_gate_kernel_sv, 1), 
            'tdg': (self._apply_tdg_kernel_sv, 1),
            'rx': (self._apply_rx_kernel_sv, 2), 
            'ry': (self._apply_ry_kernel_sv, 2),
            'rz': (self._apply_rz_kernel_sv, 2), 
            'p_gate': (self._apply_p_gate_kernel_sv, 2),
            'u3_gate': (self._apply_u3_gate_kernel_sv, 4), # (qubit, theta, phi, lambda)
            'cnot': (self._apply_cnot_kernel_sv, 2), 
            'cz': (self._apply_cz_kernel_sv, 2),
            'cp': (self._apply_cp_kernel_sv, 3), # (control, target, angle)
            'crx': (self._apply_crx_kernel_sv, 3), # (control, target, theta)
            'cry': (self._apply_cry_kernel_sv, 3), # (control, target, theta)
            'crz': (self._apply_crz_kernel_sv, 3), # (control, target, phi)
            'controlled_u': (self._apply_controlled_u_kernel_sv, 3), # (control, target, u_matrix)
            'rxx': (self._apply_rxx_kernel_sv, 3), # (q1, q2, theta)
            'ryy': (self._apply_ryy_kernel_sv, 3), # (q1, q2, theta)
            'rzz': (self._apply_rzz_kernel_sv, 3), # (q1, q2, theta)
            'toffoli': (self._apply_toffoli_kernel_sv, 3), # (c1, c2, target)
            'apply_unitary': (self._apply_apply_unitary_kernel_sv, 2) # 优化器生成的特殊指令
        }
        
        self._internal_logger.debug(f"[_StateVectorEntity.__init__(N={num_qubits})] Initialized successfully with {type(self._backend).__name__}.")
    
    @property
    def math(self):
        """提供对当前后端数学库 (math or cupy) 的访问。"""
        return self._backend.math

    @property
    def cmath(self):
        """提供对当前后端复数数学库 (cmath or cupy) 的访问。"""
        return self._backend.cmath
    
    
    
    def _execute_parallel_h_kernel(self, target_qubit: int):
        """
        [v1.5.2 刷屏问题修正版] 在主进程中编排 Hadamard 内核的并行执行。
        此版本修正了当多个 worker 同时失败时，错误信息会重复打印（刷屏）的问题。

        核心修正:
        -   **快速失败 (Fail-Fast) 机制**: 监控循环现在会在检测到第一个
            来自 worker 的致命错误报告或任何一个 worker 进程静默失败后，
            立即中断循环并进入清理阶段。
        -   **单一错误报告**: 通过 `fatal_error_from_worker` 变量，只捕获并
            记录第一个收到的 worker 错误。在最终的异常抛出阶段，只会打印
            这一个错误的详细 traceback，避免了信息冗余。
        -   **健壮的资源清理**: 无论循环是正常结束还是因错误中断，`finally`
            块都会被执行。如果检测到失败 (`all_workers_succeeded` is False)，
            会调用 `pool.terminate()` 强制、快速地终止所有仍在运行的 worker
            进程，防止它们继续占用资源或产生更多错误。

        Args:
            target_qubit (int): Hadamard 门作用的目标量子比特。

        Raises:
            MemoryError: 如果无法分配所需的共享内存。
            RuntimeError: 如果任何工作进程失败或在编排过程中发生严重错误。
        """
        # 声明对模块级全局变量的访问
        global _parallel_enabled, _num_processes, _progress_queue

        # --- 步骤 1: 前置检查与初始化 ---
        if not _parallel_enabled or _num_processes <= 0:
            self._internal_logger.warning("Parallel Hadamard kernel called, but parallelism is not enabled or configured. Falling back to serial execution.")
            self._apply_h_kernel_sv(target_qubit)
            return

        dim = 1 << self.num_qubits
        if dim == 0:
            return

        size_in_doubles = dim * 2 
        
        # --- 步骤 2: 创建并填充本次任务专用的共享内存 ---
        try:
            sv_in_shm = mp.RawArray(ctypes.c_double, size_in_doubles)
            sv_out_shm = mp.RawArray(ctypes.c_double, size_in_doubles)
            current_vec_list = self._state_vector.tolist() if hasattr(self._state_vector, 'tolist') else self._state_vector
            flat_floats = [val for c in current_vec_list for val in (c.real, c.imag)]
            sv_in_shm[:len(flat_floats)] = flat_floats
            shm_arrays_for_task = {'sv_in': sv_in_shm, 'sv_out': sv_out_shm}
        except Exception as e:
            self._internal_logger.critical(f"Failed to allocate or fill shared memory for parallel Hadamard kernel: {e}", exc_info=True)
            raise MemoryError("Shared memory allocation failed for parallel Hadamard.") from e

        # --- 步骤 3: 将总任务分割成多个微任务块 ---
        num_effective_indices = dim // 2
        num_chunks = _num_processes * 4
        chunk_size = max(1, math.ceil(num_effective_indices / num_chunks))
        tasks = [{'chunk_range': (i_start, min(i_start + chunk_size, num_effective_indices)), 'dim': dim, 'target_qubit': target_qubit} for i_start in range(0, num_effective_indices, chunk_size)]
        if not tasks:
            return

        # --- 步骤 4: 创建专用临时进程池并编排任务执行 ---
        pool_for_this_run: Optional[mp.pool.Pool] = None
        all_workers_succeeded = False
        fatal_error_from_worker = None

        try:
            # 准备传递给 `init_worker` 的参数
            root_logger = logging.getLogger()
            log_level = root_logger.level if root_logger.level != 0 else logging.INFO
            log_format = "%(asctime)s - [%(levelname)s] - (%(name)s) - [PID:%(process)d] - %(message)s"
            init_args = (log_level, log_format, _progress_queue, shm_arrays_for_task)
            
            ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
            pool_for_this_run = ctx.Pool(processes=_num_processes, initializer=init_worker, initargs=init_args)

            total_tasks = len(tasks)
            results = [pool_for_this_run.apply_async(_worker_apply_h_chunk, args=(task,)) for task in tasks]
            self._internal_logger.info(f"Asynchronously submitted {total_tasks} chunks for parallel Hadamard to {_num_processes} workers.")
            
            completed_tasks = 0
            
            # [核心修正] 带有快速失败机制的监控循环
            while completed_tasks < total_tasks:
                try:
                    # 从队列中获取消息
                    message = _progress_queue.get(timeout=0.2)
                    
                    if isinstance(message, dict) and message.get("type") == "error":
                        if fatal_error_from_worker is None:
                            fatal_error_from_worker = message
                        # 收到错误报告，立即跳出循环
                        break
                    
                    elif message == 1:
                        completed_tasks += 1
                    else:
                        self._internal_logger.warning(f"Received unknown message from progress queue: {message}")

                except mp.queues.Empty:
                    # 队列为空时，检查是否有 worker 已经静默失败
                    if any(res.ready() and not res.successful() for res in results):
                        self._internal_logger.error("A worker process has terminated unexpectedly (silent failure). Aborting.")
                        # 静默失败，也立即跳出循环
                        break 
                    time.sleep(0.05)

            # --- [核心修正] 循环结束后的统一处理 ---
            if fatal_error_from_worker:
                # 如果是因为收到错误报告而跳出循环
                raise RuntimeError("A worker process reported a fatal error.")
            
            if completed_tasks < total_tasks:
                # 如果是因为静默失败或其他原因提前跳出循环
                # 尝试从 result 对象中获取更详细的异常信息
                for res in results:
                    if res.ready() and not res.successful():
                        try:
                            res.get() # 这会重新抛出子进程中的异常
                        except Exception as worker_exc:
                            # 记录第一个能获取到的异常
                            self._internal_logger.error(f"Caught exception from a silently failed worker: {worker_exc}", exc_info=True)
                            break
                raise RuntimeError("One or more worker processes failed during execution.")

            # 如果循环正常完成
            final_results = [res.get() for res in results]
            all_workers_succeeded = all(final_results)
            
        except (KeyboardInterrupt, SystemExit):
            self._internal_logger.warning("Parallel Hadamard execution was manually interrupted by the user.")
            all_workers_succeeded = False
        except Exception as e:
            self._internal_logger.critical(f"Error during parallel Hadamard execution orchestration: {e}", exc_info=False)
            if fatal_error_from_worker:
                self._internal_logger.error(f"--- Full Traceback from Failing Worker (PID: {fatal_error_from_worker.get('pid')}) ---\n"
                                            f"{fatal_error_from_worker.get('traceback', 'No traceback available.')}")
            all_workers_succeeded = False
        finally:
            # 健壮的资源清理
            if pool_for_this_run:
                self._internal_logger.debug("Terminating the temporary parallel Hadamard pool...")
                # [核心修正] 如果检测到任何失败，就强制终止所有进程
                if not all_workers_succeeded: 
                    pool_for_this_run.terminate()
                pool_for_this_run.close()
                pool_for_this_run.join()
                self._internal_logger.debug("Temporary parallel Hadamard pool terminated.")
        
        if not all_workers_succeeded:
            raise RuntimeError("Parallel Hadamard kernel execution failed. Check logs for details.")

        # --- 步骤 5: 从共享内存反序列化结果并更新态矢量 ---
        final_sv_flat = list(sv_out_shm)
        reconstructed_complex_list = [complex(final_sv_flat[i], final_sv_flat[i + 1]) for i in range(0, size_in_doubles, 2)]

        if isinstance(self._backend, PurePythonBackend):
            self._state_vector = reconstructed_complex_list
        else:
            self._state_vector = self._backend._ensure_cupy_array(reconstructed_complex_list)
    
    
    
    
    
    
    # --- 内部优化内核: 纯Python模式下的门内核 (经过全面修正) ---
    # 这些方法是为态矢量模拟特别优化的低级内核。
    # 它们直接操作 `self._state_vector`，避免构建全局矩阵。

    def _apply_x_kernel_sv(self, target_qubit: int):
        """
        [核心修正] 在态矢量上应用 Pauli-X 门。
        """
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            raise ValueError(f"Invalid target qubit index {target_qubit}.")

        dim = 1 << self.num_qubits
        sv = self._state_vector
        flip_mask = 1 << target_qubit
        is_list_backend = isinstance(sv, list)

        for i in range(dim):
            if (i & flip_mask) == 0: # 只处理目标比特为0的索引，避免重复交换
                j = i | flip_mask
                if is_list_backend:
                    sv[i], sv[j] = sv[j], sv[i]
                else: # CuPy 后端
                    temp = sv[i].item() # 使用 .item() 保证安全
                    sv[i] = sv[j]
                    sv[j] = temp
        self._normalize()

    def _apply_y_kernel_sv(self, target_qubit: int):
        """
        [核心修正] 在态矢量上应用 Pauli-Y 门。
        """
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            raise ValueError(f"Invalid target qubit index {target_qubit}.")

        dim = 1 << self.num_qubits
        sv = self._state_vector
        target_mask = 1 << target_qubit
        is_list_backend = isinstance(sv, list)

        for i in range(dim):
            if (i & target_mask) == 0: # 只处理目标比特为0的索引
                j = i | target_mask
                # 先读取，再写入
                if is_list_backend:
                    amp0, amp1 = sv[i], sv[j]
                else: # CuPy 后端
                    amp0, amp1 = sv[i].item(), sv[j].item()
                
                sv[i] = -1j * amp1
                sv[j] =  1j * amp0
        self._normalize()

    def _apply_z_kernel_sv(self, target_qubit: int):
        """
        [内部优化内核][最终修正版] 在态矢量上应用 Pauli-Z 门。
        此操作通过给 |...1...⟩ 基态的振幅乘以 -1 来实现。
        """
        # [健壮性改进] 输入验证
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            self._internal_logger.error(f"_apply_z_kernel_sv: Invalid 'target_qubit' {target_qubit} for {self.num_qubits} qubits.")
            raise ValueError(f"Invalid target qubit index {target_qubit}.")

        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        for i in range(dim):
            if (i >> target_qubit) & 1:
                sv[i] *= -1.0
                
        self._normalize()

    def _apply_h_kernel_sv(self, target_qubit: int):
        """
        [核心修正] 在态矢量上应用 Hadamard 门 (串行版本)。
        """
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            raise ValueError(f"Invalid target qubit index {target_qubit}.")

        if (_parallel_enabled and self.num_qubits >= _core_config["PARALLEL_COMPUTATION_QUBIT_THRESHOLD"] and isinstance(self._backend, PurePythonBackend)):
            self._execute_parallel_h_kernel(target_qubit)
            self._normalize()
            return

        dim = 1 << self.num_qubits
        sv = self._state_vector
        sqrt2_inv = 1.0 / self._backend.math.sqrt(2.0)
        target_mask = 1 << target_qubit
        is_list_backend = isinstance(sv, list)

        for i in range(dim):
            if (i & target_mask) == 0: # 只处理目标比特为0的索引
                j = i | target_mask
                # 先读取，再写入
                if is_list_backend:
                    amp0, amp1 = sv[i], sv[j]
                else: # CuPy 后端
                    amp0, amp1 = sv[i].item(), sv[j].item()
                
                new_amp0 = (amp0 + amp1) * sqrt2_inv
                new_amp1 = (amp0 - amp1) * sqrt2_inv
                
                sv[i], sv[j] = new_amp0, new_amp1
        self._normalize()
    
    
    def _apply_s_kernel_sv(self, target_qubit: int):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 S (Phase) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        S 门是一个对角矩阵，其作用是给计算基矢 `|1⟩` 分量施加一个 `i` (e^(i*π/2)) 的相位，
        而保持 `|0⟩` 分量不变。

        因此，此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`，
        检查其 `target_qubit` 位是否为 `1`。如果是，则将该索引对应的振幅 `sv[i]` 乘以 `i`。

        此版本经过了全面的健壮性增强，包括：
        - 严格的输入验证。
        - 通过 `self._backend` 抽象所有数学运算，以兼容不同后端。
        - 增加了详细的日志记录。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            target_qubit (int): S 门作用的目标量子比特索引。

        Raises:
            ValueError: 如果 `target_qubit` 不是一个有效的量子比特索引。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            self._internal_logger.error(f"_apply_s_kernel_sv: Invalid 'target_qubit' {target_qubit} for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid target qubit index {target_qubit}.")

        self._internal_logger.debug(f"Applying S kernel on qubit {target_qubit}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数获取相位因子 i = e^(i*π/2)
        # 这确保了无论后端是纯Python还是CuPy，都能正确计算
        phase = self._backend.cmath.exp(1j * self._backend.math.pi / 2.0)
        
        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 检查当前索引 `i` 的 `target_qubit` 位是否为 1
            # `(i >> target_qubit) & 1` 是一个高效的位运算，用于提取特定比特的值
            if (i >> target_qubit) & 1:
                # 如果是，则将对应的振幅乘以相位因子 `i`
                sv[i] *= phase
                
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 S 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()

    def _apply_sdg_kernel_sv(self, target_qubit: int):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 S-dagger (S†) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        Sdg 门是 S 门的共轭转置，其作用是给计算基矢 `|1⟩` 分量施加一个 `-i` (e^(-i*π/2)) 的相位，
        而保持 `|0⟩` 分量不变。

        因此，此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`，
        检查其 `target_qubit` 位是否为 `1`。如果是，则将该索引对应的振幅 `sv[i]` 乘以 `-i`。

        此版本经过了全面的健壮性增强，包括：
        - 严格的输入验证。
        - 通过 `self._backend` 抽象所有数学运算，以兼容不同后端。
        - 增加了详细的日志记录。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            target_qubit (int): Sdg 门作用的目标量子比特索引。

        Raises:
            ValueError: 如果 `target_qubit` 不是一个有效的量子比特索引。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            self._internal_logger.error(f"_apply_sdg_kernel_sv: Invalid 'target_qubit' {target_qubit} for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid target qubit index {target_qubit}.")

        self._internal_logger.debug(f"Applying Sdg kernel on qubit {target_qubit}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数获取相位因子 -i = e^(-i*π/2)
        # 这确保了无论后端是纯Python还是CuPy，都能正确计算
        phase = self._backend.cmath.exp(-1j * self._backend.math.pi / 2.0)
        
        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 检查当前索引 `i` 的 `target_qubit` 位是否为 1
            # `(i >> target_qubit) & 1` 是一个高效的位运算，用于提取特定比特的值
            if (i >> target_qubit) & 1:
                # 如果是，则将对应的振幅乘以相位因子 `-i`
                sv[i] *= phase
                
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 Sdg 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()
    def _apply_t_gate_kernel_sv(self, target_qubit: int):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 T (π/8) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        T 门是一个对角矩阵，其作用是给计算基矢 `|1⟩` 分量施加一个 `e^(i*π/4)` 的相位，
        而保持 `|0⟩` 分量不变。

        因此，此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`，
        检查其 `target_qubit` 位是否为 `1`。如果是，则将该索引对应的振幅 `sv[i]` 乘以 `e^(i*π/4)`。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证。
        - 通过 `self._backend` 抽象所有数学运算，以兼容不同后端。
        - 增加了详细的日志记录。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            target_qubit (int): T 门作用的目标量子比特索引。

        Raises:
            ValueError: 如果 `target_qubit` 不是一个有效的量子比特索引。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            self._internal_logger.error(f"_apply_t_gate_kernel_sv: Invalid 'target_qubit' {target_qubit} for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid target qubit index {target_qubit}.")

        self._internal_logger.debug(f"Applying T gate kernel on qubit {target_qubit}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数获取相位因子 e^(i*π/4)
        # 这确保了无论后端是纯Python还是CuPy，都能正确计算
        phase = self._backend.cmath.exp(1j * self._backend.math.pi / 4.0)
        
        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 检查当前索引 `i` 的 `target_qubit` 位是否为 1
            # `(i >> target_qubit) & 1` 是一个高效的位运算，用于提取特定比特的值
            if (i >> target_qubit) & 1:
                # 如果是，则将对应的振幅乘以相位因子
                sv[i] *= phase
                
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 T 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()

    def _apply_tdg_kernel_sv(self, target_qubit: int):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 T-dagger (T†) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        Tdg 门是 T 门的共轭转置，其作用是给计算基矢 `|1⟩` 分量施加一个 `e^(-i*π/4)` 的相位，
        而保持 `|0⟩` 分量不变。

        因此，此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`，
        检查其 `target_qubit` 位是否为 `1`。如果是，则将该索引对应的振幅 `sv[i]` 乘以 `e^(-i*π/4)`。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证。
        - 通过 `self._backend` 抽象所有数学运算，以兼容不同后端。
        - 增加了详细的日志记录。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            target_qubit (int): Tdg 门作用的目标量子比特索引。

        Raises:
            ValueError: 如果 `target_qubit` 不是一个有效的量子比特索引。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            self._internal_logger.error(f"_apply_tdg_kernel_sv: Invalid 'target_qubit' {target_qubit} for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid target qubit index {target_qubit}.")

        self._internal_logger.debug(f"Applying Tdg kernel on qubit {target_qubit}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数获取相位因子 e^(-i*π/4)
        # 这确保了无论后端是纯Python还是CuPy，都能正确计算
        phase = self._backend.cmath.exp(-1j * self._backend.math.pi / 4.0)
        
        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 检查当前索引 `i` 的 `target_qubit` 位是否为 1
            # `(i >> target_qubit) & 1` 是一个高效的位运算，用于提取特定比特的值
            if (i >> target_qubit) & 1:
                # 如果是，则将对应的振幅乘以相位因子
                sv[i] *= phase
                
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 Tdg 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()

    
    def _apply_sx_kernel_sv(self, target_qubit: int):
        """
        [内部优化内核][原地修改逻辑修复] 在态矢量上高效地应用 sqrt(X) (SX) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        SX 门矩阵为 `[[ (1+i)/2, (1-i)/2 ], [ (1-i)/2, (1+i)/2 ]]`。
        对于每一对振幅 (amp_low, amp_high) 对应于基态 |...0...⟩ 和 |...1...⟩，
        新的振幅为：
        amp_low'  = ((1+i)/2) * amp_low + ((1-i)/2) * amp_high
        amp_high' = ((1-i)/2) * amp_low + ((1+i)/2) * amp_high

        核心实现:
        - **原地修改修复**: 循环遍历所有索引，但只在目标比特为 `0` 的索引 `i` 处
          触发计算。在计算新振幅之前，会先将 `sv[i]` 和 `sv[j]` (其中 j 是配对索引)
          都读取到临时变量中，然后再进行写入。这确保了原地修改的正确性，
          避免了数据污染。
        - **后端无关**: 通过 `is_list_backend` 和 `.item()` 调用，确保在
          PurePython 和 CuPy 后端下都能正确工作。

        Args:
            target_qubit (int): SX 门作用的目标量子比特索引。

        Raises:
            ValueError: 如果 `target_qubit` 不是一个有效的量子比特索引。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            self._internal_logger.error(f"_apply_sx_kernel_sv: Invalid 'target_qubit' {target_qubit} for {self.num_qubits} qubits.")
            raise ValueError(f"Invalid target qubit index {target_qubit}.")

        self._internal_logger.debug(f"Applying SX kernel on qubit {target_qubit}.")
        
        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 预先计算 SX 矩阵的元素
        factor = (1.0 + 1.0j) / 2.0
        anti_factor = (1.0 - 1.0j) / 2.0
        
        target_mask = 1 << target_qubit
        is_list_backend = isinstance(sv, list)

        # --- 步骤 3: 核心计算逻辑 ---
        # [BUGFIX] 使用最终正确的原地修改循环逻辑
        for i in range(dim):
            # 为了避免对同一对振幅操作两次，我们只在目标比特为 0 的索引处触发计算
            if (i & target_mask) == 0:
                j = i | target_mask # 计算与之配对的索引
                
                # [核心修复] 必须先读取两个原始振幅，再进行写入，以避免数据污染
                if is_list_backend:
                    amp0 = sv[i]
                    amp1 = sv[j]
                else: # CuPy 后端
                    amp0 = sv[i].item()
                    amp1 = sv[j].item()
                
                # 根据 SX 矩阵的定义计算新的振幅
                new_amp0 = factor * amp0 + anti_factor * amp1
                new_amp1 = anti_factor * amp0 + factor * amp1
                
                # 现在可以安全地写回态矢量
                sv[i] = new_amp0
                sv[j] = new_amp1
                    
        # --- 步骤 4: 归一化 ---
        # 尽管 SX 门是酉操作，但为了防止浮点误差累积，进行归一化是良好实践。
        self._normalize()

    def _apply_rx_kernel_sv(self, target_qubit: int, theta: float):
        """
        [核心修正] 在态矢量上应用 RX(theta) 旋转门。
        """
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            raise ValueError(f"Invalid target qubit index {target_qubit}.")
        if not isinstance(theta, (float, int)):
            raise TypeError(f"Theta must be a numeric type.")

        dim = 1 << self.num_qubits
        sv = self._state_vector
        half_theta = theta / 2.0
        c = self._backend.math.cos(half_theta)
        s_factor = -1j * self._backend.math.sin(half_theta)
        target_mask = 1 << target_qubit
        is_list_backend = isinstance(sv, list)

        for i in range(dim):
            if (i & target_mask) == 0:
                j = i | target_mask
                if is_list_backend:
                    amp0, amp1 = sv[i], sv[j]
                else: # CuPy 后端
                    amp0, amp1 = sv[i].item(), sv[j].item()
                
                new_amp0 = c * amp0 + s_factor * amp1
                new_amp1 = s_factor * amp0 + c * amp1
                
                sv[i], sv[j] = new_amp0, new_amp1
        self._normalize()
    
    def _apply_ry_kernel_sv(self, target_qubit: int, theta: float):
        """
        [核心修正] 在态矢量上应用 RY(theta) 旋转门。
        """
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            raise ValueError(f"Invalid target qubit index {target_qubit}.")
        if not isinstance(theta, (float, int)):
            raise TypeError(f"Theta must be a numeric type.")

        dim = 1 << self.num_qubits
        sv = self._state_vector
        half_theta = theta / 2.0
        c, s = self._backend.math.cos(half_theta), self._backend.math.sin(half_theta)
        target_mask = 1 << target_qubit
        is_list_backend = isinstance(sv, list)

        for i in range(dim):
            if (i & target_mask) == 0:
                j = i | target_mask
                if is_list_backend:
                    amp0, amp1 = sv[i], sv[j]
                else: # CuPy 后端
                    amp0, amp1 = sv[i].item(), sv[j].item()
                
                new_amp0 = c * amp0 - s * amp1
                new_amp1 = s * amp0 + c * amp1
                
                sv[i], sv[j] = new_amp0, new_amp1
        self._normalize()
    
    def _apply_rz_kernel_sv(self, target_qubit: int, phi: float):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 RZ(phi) 旋转门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        RZ(φ) 门是一个对角矩阵：[[e^(-iφ/2), 0], [0, e^(iφ/2)]]。
        其作用是给计算基矢 `|0⟩` 分量施加一个 `e^(-iφ/2)` 的相位，
        给计算基矢 `|1⟩` 分量施加一个 `e^(iφ/2)` 的相位。

        因此，此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`，
        检查其 `target_qubit` 位的值。如果为 `0`，则将振幅乘以 `e^(-iφ/2)`；
        如果为 `1`，则将振幅乘以 `e^(iφ/2)`。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证。
        - 通过 `self._backend` 抽象所有数学运算。
        - 增加了详细的日志记录和注释。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            target_qubit (int): RZ 门作用的目标量子比特索引。
            phi (float): 旋转角度 (单位：弧度)。

        Raises:
            ValueError: 如果 `target_qubit` 不是一个有效的量子比特索引。
            TypeError: 如果 `phi` 不是一个数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            self._internal_logger.error(f"_apply_rz_kernel_sv: Invalid 'target_qubit' {target_qubit} for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid target qubit index {target_qubit}.")
        if not isinstance(phi, (float, int)):
            self._internal_logger.error(f"_apply_rz_kernel_sv: 'phi' must be a numeric type, but got {type(phi).__name__}.")
            raise TypeError(f"Phi must be a numeric type.")

        self._internal_logger.debug(f"Applying RZ({phi:.4f}) kernel on qubit {target_qubit}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数预计算相位因子
        half_phi = phi / 2.0
        phase_0_state = self._backend.cmath.exp(-1j * half_phi)
        phase_1_state = self._backend.cmath.exp(1j * half_phi)
        
        target_mask = 1 << target_qubit

        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 检查当前索引 `i` 的 `target_qubit` 位的值
            if (i & target_mask) == 0: # 目标比特为 0
                sv[i] *= phase_0_state
            else: # 目标比特为 1
                sv[i] *= phase_1_state
                
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 RZ 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()

    def _apply_p_gate_kernel_sv(self, target_qubit: int, lambda_angle: float):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 P(lambda) 门 (Phase Gate)。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        P(λ) 门是一个对角矩阵：[[1, 0], [0, e^(iλ)]]。
        其作用是给计算基矢 `|1⟩` 分量施加一个 `e^(iλ)` 的相位，
        而保持 `|0⟩` 分量不变。

        因此，此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`，
        检查其 `target_qubit` 位是否为 `1`。如果是，则将该索引对应的振幅 `sv[i]` 乘以 `e^(iλ)`。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证。
        - 通过 `self._backend` 抽象所有数学运算。
        - 增加了详细的日志记录和注释。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            target_qubit (int): P 门作用的目标量子比特索引。
            lambda_angle (float): 旋转角度 (单位：弧度)。

        Raises:
            ValueError: 如果 `target_qubit` 不是一个有效的量子比特索引。
            TypeError: 如果 `lambda_angle` 不是一个数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(target_qubit, int) or not (0 <= target_qubit < self.num_qubits):
            self._internal_logger.error(f"_apply_p_gate_kernel_sv: Invalid 'target_qubit' {target_qubit} for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid target qubit index {target_qubit}.")
        if not isinstance(lambda_angle, (float, int)):
            self._internal_logger.error(f"_apply_p_gate_kernel_sv: 'lambda_angle' must be a numeric type, but got {type(lambda_angle).__name__}.")
            raise TypeError(f"Lambda angle must be a numeric type.")

        self._internal_logger.debug(f"Applying P({lambda_angle:.4f}) kernel on qubit {target_qubit}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数获取相位因子 e^(iλ)
        phase = self._backend.cmath.exp(1j * lambda_angle)
        
        target_mask = 1 << target_qubit

        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 检查当前索引 `i` 的 `target_qubit` 位是否为 1
            if (i & target_mask) != 0: # `& target_mask` is slightly faster than `>>` if mask is precomputed
                # 如果是，则将对应的振幅乘以相位因子
                sv[i] *= phase
                
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 P 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()


    def _apply_u3_gate_kernel_sv(self, qubit_index: int, theta: float, phi: float, lambda_angle: float):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用通用的 U3 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        U3(θ, φ, λ) 门是单比特酉操作的一般形式，其矩阵为：
        [[cos(θ/2), -e^(iλ)sin(θ/2)], [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]。

        此内核的逻辑是遍历所有基态对 `|...0...⟩` 和 `|...1...⟩`（其中 `...` 部分相同），
        并根据 U3 矩阵的定义更新它们的振幅。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证。
        - 通过 `self._backend` 抽象所有数学运算。
        - 增加了对 CuPy 后端的特别处理（使用 `.item()`），以确保原地修改的正确性。
        - 增加了详细的日志记录和注释。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            qubit_index (int): U3 门作用的目标量子比特索引。
            theta (float): 旋转角度 θ (单位：弧度)。
            phi (float): 旋转角度 φ (单位：弧度)。
            lambda_angle (float): 旋转角度 λ (单位：弧度)。

        Raises:
            ValueError: 如果 `qubit_index` 不是一个有效的量子比特索引。
            TypeError: 如果任何角度参数不是数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"_apply_u3_gate_kernel_sv: Invalid 'qubit_index' {qubit_index} for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        if not all(isinstance(a, (float, int)) for a in [theta, phi, lambda_angle]):
            self._internal_logger.error(f"_apply_u3_gate_kernel_sv: All angles must be numeric. Got types: ({type(theta).__name__}, {type(phi).__name__}, {type(lambda_angle).__name__}).")
            raise TypeError(f"All angles (theta, phi, lambda_angle) must be numeric.")

        self._internal_logger.debug(f"Applying U3({theta:.4f}, {phi:.4f}, {lambda_angle:.4f}) kernel on qubit {qubit_index}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数预计算 U3 矩阵的元素
        c = self._backend.math.cos(theta / 2.0)
        s = self._backend.math.sin(theta / 2.0)
        u00 = c
        u01 = -self._backend.cmath.exp(1j * lambda_angle) * s
        u10 = self._backend.cmath.exp(1j * phi) * s
        u11 = self._backend.cmath.exp(1j * (phi + lambda_angle)) * c
        
        target_mask = 1 << qubit_index
        is_list_backend = isinstance(sv, list)

        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量，每次处理一对振幅
        for i in range(dim):
            # 为了避免重复计算，我们只处理那些目标比特为 0 的索引
            if (i & target_mask) == 0:
                # i 对应 |...0...⟩ 基态
                # j 对应 |...1...⟩ 基态
                j = i | target_mask
                
                # [健 robuste性改进] 对不同后端进行处理
                if is_list_backend:
                    # 对于纯Python列表，直接访问
                    amp0 = sv[i]
                    amp1 = sv[j]
                else: # CuPy 后端
                    # [核心修正] 使用 .item() 将 CuPy 0维数组转换为独立的 Python 标量
                    amp0 = sv[i].item()
                    amp1 = sv[j].item()
                
                # 根据 U3 矩阵的定义计算新的振幅
                new_amp0 = u00 * amp0 + u01 * amp1
                new_amp1 = u10 * amp0 + u11 * amp1
                
                # 将计算出的新振幅写回态矢量
                if is_list_backend:
                    sv[i] = new_amp0
                    sv[j] = new_amp1
                else: # CuPy 后端
                    sv[i] = new_amp0
                    sv[j] = new_amp1
                    
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 U3 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()

    def _apply_cnot_kernel_sv(self, control_qubit: int, target_qubit: int):
        """
        [内部优化内核][最终修正版] 在态矢量上应用 CNOT 门。
        此版本通过使用 .item() 方法，修复了在 CuPy 后端下因原地修改
        可能导致的 CUDA 内存访问错误。
        """
        # [健壮性改进] 输入验证
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control_qubit, target_qubit]):
            self._internal_logger.error(f"_apply_cnot_kernel_sv: Invalid control/target qubits ({control_qubit}, {target_qubit}) for {self.num_qubits} qubits.")
            raise ValueError(f"Invalid control or target qubit index.")
        if control_qubit == target_qubit:
            self._internal_logger.error(f"_apply_cnot_kernel_sv: Control qubit ({control_qubit}) cannot be the same as target qubit ({target_qubit}).")
            raise ValueError("Control and target qubits cannot be the same.")

        dim = 1 << self.num_qubits
        sv = self._state_vector
        control_mask = 1 << control_qubit
        target_mask = 1 << target_qubit
        is_list_backend = isinstance(sv, list)

        for i in range(dim):
            # CNOT 操作只在控制比特为 1 时触发
            # 并且，为了避免重复交换，我们只处理目标比特为 0 的情况
            if (i & control_mask) != 0 and (i & target_mask) == 0:
                # i 对应 |...1_c...0_t...⟩ 基态
                # j 对应 |...1_c...1_t...⟩ 基态
                j = i | target_mask
                
                # 执行交换
                if is_list_backend:
                    sv[i], sv[j] = sv[j], sv[i]
                else: # CuPy 后端
                    temp = sv[i].item()
                    sv[i] = sv[j]
                    sv[j] = temp
                    
        self._normalize()

    def _apply_cz_kernel_sv(self, control_qubit: int, target_qubit: int):
        """
        [内部优化内核][最终修正版] 在态矢量上应用 Controlled-Z (CZ) 门。
        此操作通过给 |...1_c...1_t...⟩ 基态的振幅乘以 -1 来实现。
        """
        # [健壮性改进] 输入验证
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control_qubit, target_qubit]):
            self._internal_logger.error(f"_apply_cz_kernel_sv: Invalid control/target qubits ({control_qubit}, {target_qubit}) for {self.num_qubits} qubits.")
            raise ValueError(f"Invalid control or target qubit index.")
        if control_qubit == target_qubit:
            self._internal_logger.error(f"_apply_cz_kernel_sv: Control qubit ({control_qubit}) cannot be the same as target qubit ({target_qubit}).")
            raise ValueError("Control and target qubits cannot be the same.")

        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        mask = (1 << control_qubit) | (1 << target_qubit)
        
        for i in range(dim):
            if (i & mask) == mask: # 两个比特都为 1
                sv[i] *= -1.0
                
        self._normalize()

    def _apply_toffoli_kernel_sv(self, control_1: int, control_2: int, target: int):
        """
        [核心修正] 在态矢量上应用 Toffoli (CCNOT) 门。
        """
        qubits = [control_1, control_2, target]
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits):
            raise ValueError(f"Invalid qubit index for Toffoli gate.")
        if len(set(qubits)) != 3:
            raise ValueError("All qubits for Toffoli gate must be distinct.")

        dim = 1 << self.num_qubits
        sv = self._state_vector
        control_mask = (1 << control_1) | (1 << control_2)
        target_mask = 1 << target
        is_list_backend = isinstance(sv, list)

        for i in range(dim):
            # [BUGFIX] 正确的循环条件
            if (i & control_mask) == control_mask and (i & target_mask) == 0:
                j = i | target_mask
                if is_list_backend:
                    sv[i], sv[j] = sv[j], sv[i]
                else: # CuPy 后端
                    temp = sv[i].item()
                    sv[i] = sv[j]
                    sv[j] = temp
        self._normalize()
    def _apply_cp_kernel_sv(self, control: int, target: int, angle: float):
        """
        [内部优化内核][最终修正版] 在态矢量上应用 Controlled-Phase (CP) 门。
        此操作通过给 |...1_c...1_t...⟩ 基态的振幅乘以 e^(i*angle) 来实现。
        """
        # [健壮性改进] 输入验证
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"_apply_cp_kernel_sv: Invalid control/target qubits ({control}, {target}) for {self.num_qubits} qubits.")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"_apply_cp_kernel_sv: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        if not isinstance(angle, (float, int)):
            self._internal_logger.error(f"_apply_cp_kernel_sv: 'angle' must be numeric, got {type(angle).__name__}.")
            raise TypeError(f"Angle must be a numeric type.")

        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        mask = (1 << control) | (1 << target)
        
        phase = self.cmath.exp(1j * angle)
        
        for i in range(dim):
            if (i & mask) == mask:
                sv[i] *= phase
                
        self._normalize()

    def _apply_crx_kernel_sv(self, control: int, target: int, theta: float):
        """
        [内部优化内核][原地修改逻辑修复] 在态矢量上高效地应用 Controlled-RX(theta) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        CRX 门仅在控制比特 `control` 处于 `|1⟩` 态时，在目标比特 `target` 上应用一个 RX(θ) 旋转。

        核心实现:
        - **原地修改修复**: 循环遍历所有索引，但只在目标比特为 `0` 的索引 `i` 处
          触发计算。在计算新振幅之前，会先将 `sv[i]` 和 `sv[j]` (其中 j 是配对索引)
          都读取到临时变量中，然后再进行写入。
        - **集成控制逻辑**: 在对一对振幅进行旋转之前，通过位掩码检查控制位是否为 `1`。
        - **后端无关**: 通过 `is_list_backend`、`.item()` 调用以及 `self._backend.math`，
          确保在 PurePython 和 CuPy 后端下都能正确工作。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            theta (float): 旋转角度 (单位：弧度)。

        Raises:
            ValueError: 如果 `control` 或 `target` 无效，或它们相同。
            TypeError: 如果 `theta` 不是一个数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"_apply_crx_kernel_sv: Invalid control/target qubits ({control}, {target}) for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"_apply_crx_kernel_sv: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"_apply_crx_kernel_sv: 'theta' must be a numeric type, but got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")

        self._internal_logger.debug(f"Applying CRX({theta:.4f}) kernel on control={control}, target={target}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数预计算 RX 矩阵的元素
        half_theta = theta / 2.0
        c = self._backend.math.cos(half_theta)
        s_factor = -1j * self._backend.math.sin(half_theta)
        
        control_mask = 1 << control
        target_mask = 1 << target
        is_list_backend = isinstance(sv, list)

        # --- 步骤 3: 核心计算逻辑 ---
        # [BUGFIX] 使用最终正确的原地修改循环逻辑
        for i in range(dim):
            # 为了避免对同一对振幅操作两次，我们只在目标比特为 0 的索引处触发计算
            if (i & target_mask) == 0:
                # 只有在控制位为 1 的子空间内才执行旋转
                if (i & control_mask) != 0:
                    j = i | target_mask # 计算与之配对的索引

                    # [核心修复] 必须先读取两个原始振幅，再进行写入
                    if is_list_backend:
                        amp0 = sv[i]
                        amp1 = sv[j]
                    else: # CuPy 后端
                        amp0 = sv[i].item()
                        amp1 = sv[j].item()
                    
                    # 根据 RX 矩阵的定义计算新的振幅
                    new_amp0 = c * amp0 + s_factor * amp1
                    new_amp1 = s_factor * amp0 + c * amp1
                    
                    # 现在可以安全地写回态矢量
                    sv[i] = new_amp0
                    sv[j] = new_amp1
                        
        # --- 步骤 4: 归一化 ---
        # 尽管 CRX 门是酉操作，但为了防止浮点误差累积，进行归一化是良好实践。
        self._normalize()
    
    
    def _apply_cry_kernel_sv(self, control: int, target: int, theta: float):
        """
        [内部优化内核][原地修改逻辑修复] 在态矢量上高效地应用 Controlled-RY(theta) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        CRY 门仅在控制比特 `control` 处于 `|1⟩` 态时，在目标比特 `target` 上应用一个 RY(θ) 旋转。

        核心实现:
        - **原地修改修复**: 循环遍历所有索引，但只在目标比特为 `0` 的索引 `i` 处
          触发计算。在计算新振幅之前，会先将 `sv[i]` 和 `sv[j]` (其中 j 是配对索引)
          都读取到临时变量中，然后再进行写入。
        - **集成控制逻辑**: 在对一对振幅进行旋转之前，通过位掩码检查控制位是否为 `1`。
        - **后端无关**: 通过 `is_list_backend`、`.item()` 调用以及 `self._backend.math`，
          确保在 PurePython 和 CuPy 后端下都能正确工作。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            theta (float): 旋转角度 (单位：弧度)。

        Raises:
            ValueError: 如果 `control` 或 `target` 无效，或它们相同。
            TypeError: 如果 `theta` 不是一个数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"_apply_cry_kernel_sv: Invalid control/target qubits ({control}, {target}) for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"_apply_cry_kernel_sv: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"_apply_cry_kernel_sv: 'theta' must be a numeric type, but got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")

        self._internal_logger.debug(f"Applying CRY({theta:.4f}) kernel on control={control}, target={target}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数预计算 RY 矩阵的元素
        half_theta = theta / 2.0
        c, s = self._backend.math.cos(half_theta), self._backend.math.sin(half_theta)
        
        control_mask = 1 << control
        target_mask = 1 << target
        is_list_backend = isinstance(sv, list)

        # --- 步骤 3: 核心计算逻辑 ---
        # [BUGFIX] 使用最终正确的原地修改循环逻辑
        for i in range(dim):
            # 为了避免对同一对振幅操作两次，我们只在目标比特为 0 的索引处触发计算
            if (i & target_mask) == 0:
                # 只有在控制位为 1 的子空间内才执行旋转
                if (i & control_mask) != 0:
                    j = i | target_mask # 计算与之配对的索引

                    # [核心修复] 必须先读取两个原始振幅，再进行写入
                    if is_list_backend:
                        amp0 = sv[i]
                        amp1 = sv[j]
                    else: # CuPy 后端
                        amp0 = sv[i].item()
                        amp1 = sv[j].item()
                    
                    # 根据 RY 矩阵的定义计算新的振幅
                    new_amp0 = c * amp0 - s * amp1
                    new_amp1 = s * amp0 + c * amp1
                    
                    # 现在可以安全地写回态矢量
                    sv[i] = new_amp0
                    sv[j] = new_amp1
                        
        # --- 步骤 4: 归一化 ---
        # 尽管 CRY 门是酉操作，但为了防止浮点误差累积，进行归一化是良好实践。
        self._normalize()
    
    
    
    def _apply_crz_kernel_sv(self, control: int, target: int, phi: float):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 Controlled-RZ(phi) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        CRZ 门仅在控制比特 `control` 处于 `|1⟩` 态时，在目标比特 `target` 上应用一个 RZ(φ) 旋转。
        RZ(φ) 门本身是一个对角矩阵，因此 CRZ 也是一个对角矩阵。

        此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`：
        1. 检查 `control` 比特是否为 `1`。
        2. 如果是，则根据 `target` 比特的值（0 或 1），将振幅乘以相应的相位因子
           `e^(-iφ/2)` 或 `e^(iφ/2)`。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证。
        - 通过 `self._backend` 抽象所有数学运算。
        - 增加了详细的日志记录和注释。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            phi (float): 旋转角度 (单位：弧度)。

        Raises:
            ValueError: 如果 `control` 或 `target` 无效，或它们相同。
            TypeError: 如果 `phi` 不是一个数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"_apply_crz_kernel_sv: Invalid control/target qubits ({control}, {target}) for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"_apply_crz_kernel_sv: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        if not isinstance(phi, (float, int)):
            self._internal_logger.error(f"_apply_crz_kernel_sv: 'phi' must be a numeric type, but got {type(phi).__name__}.")
            raise TypeError(f"Phi must be a numeric type.")

        self._internal_logger.debug(f"Applying CRZ({phi:.4f}) kernel on control={control}, target={target}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数预计算相位因子
        half_phi = phi / 2.0
        phase_minus = self._backend.cmath.exp(-1j * half_phi) # 用于目标比特为 |0⟩
        phase_plus = self._backend.cmath.exp(1j * half_phi)  # 用于目标比特为 |1⟩
        
        control_mask = 1 << control
        target_mask = 1 << target

        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 检查控制比特是否为 1
            if (i & control_mask) != 0:
                # 如果是，则根据目标比特的值施加相位
                if (i & target_mask) == 0: # 目标比特为 0
                    sv[i] *= phase_minus
                else: # 目标比特为 1
                    sv[i] *= phase_plus
                    
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 CRZ 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()
    def _apply_controlled_u_kernel_sv(self, control: int, target: int, u_matrix: List[List[complex]], **kwargs: Any):
        """
        [内部优化内核][原地修改逻辑修复] 在态矢量上高效地应用通用的 Controlled-U 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        Controlled-U 门仅在控制比特 `control` 处于 `|1⟩` 态时，在目标比特 `target` 上
        应用一个指定的 2x2 酉矩阵 `u_matrix`。

        核心实现:
        - **原地修改修复**: 循环遍历所有索引，但只在目标比特为 `0` 的索引 `i` 处
          触发计算。在计算新振幅之前，会先将 `sv[i]` 和 `sv[j]` (其中 j 是配对索引)
          都读取到临时变量中，然后再进行写入。
        - **集成控制逻辑**: 在对一对振幅进行变换之前，通过位掩码检查控制位是否为 `1`。
        - **后端无关**: 通过 `is_list_backend` 和 `.item()` 调用，确保在
          PurePython 和 CuPy 后端下都能正确工作。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            u_matrix (List[List[complex]]): 要应用的 2x2 单比特酉矩阵。

        Raises:
            ValueError: 如果 `control` 或 `target` 无效、它们相同，或 `u_matrix` 格式不正确。
            TypeError: 如果输入参数类型不正确。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"_apply_controlled_u_kernel_sv: Invalid control/target qubits ({control}, {target}) for {self.num_qubits} qubits.")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"_apply_controlled_u_kernel_sv: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        
        # 验证 u_matrix 的形状和内容
        if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
                isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
                isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2 and
                all(isinstance(el, (complex, float, int)) for row in u_matrix for el in row)):
            self._internal_logger.error(f"_apply_controlled_u_kernel_sv: 'u_matrix' must be a 2x2 nested list of complex numbers. Got {u_matrix}.")
            raise ValueError("`u_matrix` for controlled_u must be a 2x2 nested list of complex numbers.")

        self._internal_logger.debug(f"Applying Controlled-U kernel on control={control}, target={target}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 从 u_matrix 中提取元素并确保它们是复数
        u00, u01 = complex(u_matrix[0][0]), complex(u_matrix[0][1])
        u10, u11 = complex(u_matrix[1][0]), complex(u_matrix[1][1])
        
        control_mask = 1 << control
        target_mask = 1 << target
        
        is_list_backend = isinstance(sv, list)

        # --- 步骤 3: 核心计算逻辑 ---
        # [BUGFIX] 使用最终正确的原地修改循环逻辑
        for i in range(dim):
            # 为了避免对同一对振幅操作两次，我们只在目标比特为 0 的索引处触发计算
            if (i & target_mask) == 0:
                # 只有在控制位为 1 的子空间内才执行变换
                if (i & control_mask) != 0:
                    j = i | target_mask # 计算与之配对的索引

                    # [核心修复] 必须先读取两个原始振幅，再进行写入
                    if is_list_backend:
                        amp0 = sv[i]
                        amp1 = sv[j]
                    else: # CuPy 后端
                        amp0 = sv[i].item()
                        amp1 = sv[j].item()
                    
                    # 根据 u_matrix 的定义计算新的振幅
                    new_amp0 = u00 * amp0 + u01 * amp1
                    new_amp1 = u10 * amp0 + u11 * amp1
                    
                    # 现在可以安全地写回态矢量
                    sv[i] = new_amp0
                    sv[j] = new_amp1
                        
        # --- 步骤 4: 归一化 ---
        # 尽管 Controlled-U 门是酉操作，但为了防止浮点误差累积，进行归一化是良好实践。
        self._normalize()
    
    
    
    def _apply_rxx_kernel_sv(self, qubit1: int, qubit2: int, theta: float):
        """
        [内部优化内核][原地修改逻辑修复] 在态矢量上高效地应用 RXX(theta) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        RXX(θ) 门在由 `|00⟩, |11⟩` 张成的子空间和由 `|01⟩, |10⟩` 张成的子空间上
        分别进行旋转。其作用可以看作两个独立的 2x2 旋转：
        - On {|00⟩, |11⟩}: [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
        - On {|01⟩, |10⟩}: [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]

        核心实现:
        - **原地修改修复**: 循环逻辑经过修正，只遍历 `dim // 4` 次。
          每次迭代处理一个由四个相关振幅组成的“四元组”，这四个振幅的索引
          仅在 `qubit1` 和 `qubit2` 位上不同。通过这种方式确保每个振幅只被
          访问和修改一次，避免了数据污染。
        - **后端无关**: 通过 `is_list_backend`、`.item()` 调用以及 `self._backend.math`，
          确保在 PurePython 和 CuPy 后端下都能正确工作。

        Args:
            qubit1 (int): 第一个目标量子比特的索引。
            qubit2 (int): 第二个目标量子比特的索引。
            theta (float): 旋转角度 (单位：弧度)。

        Raises:
            ValueError: 如果量子比特索引无效或相同。
            TypeError: 如果 `theta` 不是一个数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"_apply_rxx_kernel_sv: Invalid qubits ({qubit1}, {qubit2}) for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid qubit index for RXX gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"_apply_rxx_kernel_sv: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for RXX gate cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"_apply_rxx_kernel_sv: 'theta' must be a numeric type, but got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")

        self._internal_logger.debug(f"Applying RXX({theta:.4f}) kernel on qubits ({qubit1}, {qubit2}).")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数预计算旋转矩阵的元素
        half_theta = theta / 2.0
        c = self._backend.math.cos(half_theta)
        s_factor = -1j * self._backend.math.sin(half_theta)
        
        mask1 = 1 << qubit1
        mask2 = 1 << qubit2
        is_list_backend = isinstance(sv, list)

        # --- 步骤 3: 核心计算逻辑 ---
        # [BUGFIX] 采用正确的原地修改循环逻辑，遍历四分之一的空间
        # 每次迭代处理一个由四个相关振幅组成的“四元组”
        for i in range(dim // 4):
            # 构造一个与两个目标比特都无关的基础索引。
            # 'i' 遍历的是所有旁观者比特的组合。
            base_idx = i
            # `pos1` 追踪在 `base_idx` 中的比特位置，`pos2` 构建旁观者比特的掩码
            pos1, pos2 = 0, 0
            for j in range(self.num_qubits):
                if j != qubit1 and j != qubit2: # 如果是旁观者比特
                    if (base_idx >> pos1) & 1:
                        pos2 |= (1 << j)
                    pos1 += 1
            
            # `pos2` 现在是旁观者比特的掩码，它构成了四元组索引的基础
            idx00 = pos2               # 对应 |...0...0...⟩ (q2=0, q1=0)
            idx01 = pos2 | mask2       # 对应 |...0...1...⟩ (q2=1, q1=0)
            idx10 = pos2 | mask1       # 对应 |...1...0...⟩ (q2=0, q1=1)
            idx11 = pos2 | mask1 | mask2 # 对应 |...1...1...⟩ (q2=1, q1=1)

            # [核心修复] 先读取四元组的所有原始振幅
            if is_list_backend:
                amp00, amp01 = sv[idx00], sv[idx01]
                amp10, amp11 = sv[idx10], sv[idx11]
            else: # CuPy 后端
                amp00, amp01 = sv[idx00].item(), sv[idx01].item()
                amp10, amp11 = sv[idx10].item(), sv[idx11].item()

            # 根据 RXX 旋转矩阵的定义计算新的振幅
            # RXX 在 {|00⟩, |11⟩} 和 {|01⟩, |10⟩} 两个子空间上分别作用
            new_amp00 = c * amp00 + s_factor * amp11
            new_amp11 = s_factor * amp00 + c * amp11
            
            new_amp01 = c * amp01 + s_factor * amp10
            new_amp10 = s_factor * amp01 + c * amp10
            
            # 现在可以安全地将所有新振幅一次性写回态矢量
            if is_list_backend:
                sv[idx00], sv[idx01] = new_amp00, new_amp01
                sv[idx10], sv[idx11] = new_amp10, new_amp11
            else: # CuPy 后端
                sv[idx00], sv[idx01] = new_amp00, new_amp01
                sv[idx10], sv[idx11] = new_amp10, new_amp11
                    
        # --- 步骤 4: 归一化 ---
        # 尽管 RXX 门是酉操作，但为了防止浮点误差累积，进行归一化是良好实践。
        self._normalize()
    
    
    def _apply_ryy_kernel_sv(self, qubit1: int, qubit2: int, theta: float):
        """
        [内部优化内核][原地修改逻辑修复] 在态矢量上高效地应用 RYY(theta) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        RYY(θ) 门在由 `|00⟩, |11⟩` 张成的子空间和由 `|01⟩, |10⟩` 张成的子空间上
        分别进行旋转。其作用可以看作两个独立的 2x2 旋转：
        - On {|00⟩, |11⟩}: [[cos(θ/2),  i*sin(θ/2)], [ i*sin(θ/2), cos(θ/2)]]
        - On {|01⟩, |10⟩}: [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]

        核心实现:
        - **原地修改修复**: 循环逻辑经过修正，只遍历 `dim // 4` 次。
          每次迭代处理一个由四个相关振幅组成的“四元组”，这四个振幅的索引
          仅在 `qubit1` 和 `qubit2` 位上不同。通过这种方式确保每个振幅只被
          访问和修改一次，避免了数据污染。
        - **后端无关**: 通过 `is_list_backend`、`.item()` 调用以及 `self._backend.math`，
          确保在 PurePython 和 CuPy 后端下都能正确工作。

        Args:
            qubit1 (int): 第一个目标量子比特的索引。
            qubit2 (int): 第二个目标量子比特的索引。
            theta (float): 旋转角度 (单位：弧度)。

        Raises:
            ValueError: 如果量子比特索引无效或相同。
            TypeError: 如果 `theta` 不是一个数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"_apply_ryy_kernel_sv: Invalid qubits ({qubit1}, {qubit2}) for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid qubit index for RYY gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"_apply_ryy_kernel_sv: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for RYY gate cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"_apply_ryy_kernel_sv: 'theta' must be a numeric type, but got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")

        self._internal_logger.debug(f"Applying RYY({theta:.4f}) kernel on qubits ({qubit1}, {qubit2}).")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数预计算旋转矩阵的元素
        half_theta = theta / 2.0
        c = self._backend.math.cos(half_theta)
        s = self._backend.math.sin(half_theta)
        s_factor_pos = 1j * s
        s_factor_neg = -1j * s
        
        mask1 = 1 << qubit1
        mask2 = 1 << qubit2
        is_list_backend = isinstance(sv, list)
        
        # --- 步骤 3: 核心计算逻辑 ---
        # [BUGFIX] 采用正确的原地修改循环逻辑，遍历四分之一的空间
        # 每次迭代处理一个由四个相关振幅组成的“四元组”
        for i in range(dim // 4):
            # 构造一个与两个目标比特都无关的基础索引。
            base_idx = i
            pos1, pos2 = 0, 0
            for j in range(self.num_qubits):
                if j != qubit1 and j != qubit2: # 如果是旁观者比特
                    if (base_idx >> pos1) & 1:
                        pos2 |= (1 << j)
                    pos1 += 1

            # `pos2` 现在是旁观者比特的掩码，它构成了四元组索引的基础
            idx00 = pos2               # 对应 |...0...0...⟩
            idx01 = pos2 | mask2       # 对应 |...0...1...⟩
            idx10 = pos2 | mask1       # 对应 |...1...0...⟩
            idx11 = pos2 | mask1 | mask2 # 对应 |...1...1...⟩

            # [核心修复] 先读取四元组的所有原始振幅
            if is_list_backend:
                amp00, amp01 = sv[idx00], sv[idx01]
                amp10, amp11 = sv[idx10], sv[idx11]
            else: # CuPy 后端
                amp00, amp01 = sv[idx00].item(), sv[idx01].item()
                amp10, amp11 = sv[idx10].item(), sv[idx11].item()

            # 根据 RYY 旋转矩阵的定义计算新的振幅
            # RYY 在 {|00⟩, |11⟩} 和 {|01⟩, |10⟩} 两个子空间上分别作用
            new_amp00 = c * amp00 + s_factor_pos * amp11
            new_amp11 = s_factor_pos * amp00 + c * amp11
            
            new_amp01 = c * amp01 + s_factor_neg * amp10
            new_amp10 = s_factor_neg * amp01 + c * amp10

            # 现在可以安全地将所有新振幅一次性写回态矢量
            if is_list_backend:
                sv[idx00], sv[idx01] = new_amp00, new_amp01
                sv[idx10], sv[idx11] = new_amp10, new_amp11
            else: # CuPy 后端
                sv[idx00], sv[idx01] = new_amp00, new_amp01
                sv[idx10], sv[idx11] = new_amp10, new_amp11
                    
        # --- 步骤 4: 归一化 ---
        # 尽管 RYY 门是酉操作，但为了防止浮点误差累积，进行归一化是良好实践。
        self._normalize()
    
    def _apply_rzz_kernel_sv(self, qubit1: int, qubit2: int, theta: float):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 RZZ(theta) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        RZZ(θ) 门是一个对角矩阵，其作用是根据两个目标比特的奇偶性（Parity）施加一个相位。
        - 如果两个比特相同（`|00⟩` 或 `|11⟩`，偶数奇偶性），则施加 `e^(-iθ/2)` 相位。
        - 如果两个比特不同（`|01⟩` 或 `|10⟩`，奇数奇偶性），则施加 `e^(iθ/2)` 相位。

        因此，此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`，
        检查其 `qubit1` 和 `qubit2` 位的值，并根据它们的奇偶性将振幅 `sv[i]` 乘以对应的相位因子。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证。
        - 通过 `self._backend` 抽象所有数学运算。
        - 增加了详细的日志记录和注释。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            qubit1 (int): 第一个目标量子比特的索引。
            qubit2 (int): 第二个目标量子比特的索引。
            theta (float): 旋转角度 (单位：弧度)。

        Raises:
            ValueError: 如果量子比特索引无效或相同。
            TypeError: 如果 `theta` 不是一个数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"_apply_rzz_kernel_sv: Invalid qubits ({qubit1}, {qubit2}) for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid qubit index for RZZ gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"_apply_rzz_kernel_sv: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for RZZ gate cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"_apply_rzz_kernel_sv: 'theta' must be a numeric type, but got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")

        self._internal_logger.debug(f"Applying RZZ({theta:.4f}) kernel on qubits ({qubit1}, {qubit2}).")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用后端抽象的数学函数预计算相位因子
        half_theta = theta / 2.0
        phase_even_parity = self._backend.cmath.exp(-1j * half_theta) # 用于 |00⟩ 和 |11⟩ (偶数宇称)
        phase_odd_parity = self._backend.cmath.exp(1j * half_theta)   # 用于 |01⟩ 和 |10⟩ (奇数宇称)
        
        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 提取两个目标比特的值
            bit1 = (i >> qubit1) & 1
            bit2 = (i >> qubit2) & 1
            
            # 根据奇偶性（XOR）施加相位
            if bit1 == bit2: # 偶数奇偶性 (0 XOR 0 = 0, 1 XOR 1 = 0)
                sv[i] *= phase_even_parity
            else: # 奇数奇偶性 (0 XOR 1 = 1, 1 XOR 0 = 1)
                sv[i] *= phase_odd_parity
                
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 RZZ 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()

    # --- 高级多控制门内核 (直接在态矢量上操作) ---
    def _apply_mcz_kernel_sv(self, controls: List[int], target: int):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 Multi-Controlled-Z (MCZ) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        MCZ 门是一个对角矩阵，其作用是：当且仅当所有控制比特 `controls` 和目标比特 `target`
        都处于 `|1⟩` 态时，给该计算基矢的振幅施加一个 `-1` 的相位。

        因此，此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`，
        检查其是否同时满足所有 `controls` 比特为 `1` 且 `target` 比特为 `1`。
        如果是，则将该索引对应的振幅 `sv[i]` 乘以 `-1`。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证，确保所有比特索引有效且不重叠。
        - 使用高效的位掩码技术进行条件检查。
        - 增加了详细的日志记录和注释。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            controls (List[int]): 控制量子比特的索引列表。
            target (int): 目标量子比特的索引。

        Raises:
            ValueError: 如果 `controls` 或 `target` 包含无效或重叠的量子比特索引。
            TypeError: 如果 `controls` 不是一个列表或 `target` 不是一个整数。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"_apply_mcz_kernel_sv: Invalid 'controls' list {controls}. Must be a list of valid qubit indices.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"_apply_mcz_kernel_sv: Invalid 'target' qubit {target}. Must be a valid qubit index.")
            raise ValueError(f"Invalid target qubit index {target}.")
        if target in controls:
            self._internal_logger.error(f"_apply_mcz_kernel_sv: Target qubit ({target}) cannot be in the controls list ({controls}).")
            raise ValueError("Target qubit for MCZ gate cannot be in the controls list.")
        
        # 检查是否有重复的控制比特
        if len(set(controls)) != len(controls):
             self._internal_logger.error(f"_apply_mcz_kernel_sv: 'controls' list contains duplicate qubit indices: {controls}.")
             raise ValueError("Controls list for MCZ gate must not contain duplicate qubit indices.")

        self._internal_logger.debug(f"Applying MCZ kernel on controls={controls}, target={target}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用位掩码技术高效检查条件。
        # 创建一个掩码，其中所有控制比特和目标比特对应的位都为 1。
        full_mask = sum(1 << q for q in controls) | (1 << target)
        
        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 检查当前索引 `i` 是否与 `full_mask` 完全匹配。
            # 这等效于检查 `i` 中所有控制位和目标位是否都为 1。
            if (i & full_mask) == full_mask:
                # 如果是，则将对应的振幅乘以 -1
                sv[i] *= -1.0
                
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 MCZ 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()

    def _apply_mcx_kernel_sv(self, controls: List[int], target: int):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 Multi-Controlled-X (MCX) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        MCX 门（也称为 n-Toffoli 门）的作用是：当且仅当所有控制比特 `controls` 都处于 `|1⟩` 态时，
        在目标比特 `target` 上应用一个 Pauli-X 门（翻转）。

        此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`：
        1. 检查 `controls` 比特是否都为 `1`。
        2. 如果是，则找到与 `i` 对应的、目标比特翻转的索引 `j`，并交换它们的振幅。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证，确保所有比特索引有效且不重叠。
        - 使用高效的位掩码技术进行条件检查。
        - 增加了对 CuPy 后端的特别处理（使用 `.item()`），以确保原地修改的正确性。
        - 增加了详细的日志记录和注释。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            controls (List[int]): 控制量子比特的索引列表。
            target (int): 目标量子比特的索引。

        Raises:
            ValueError: 如果 `controls` 或 `target` 包含无效或重叠的量子比特索引。
            TypeError: 如果 `controls` 不是一个列表或 `target` 不是一个整数。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"_apply_mcx_kernel_sv: Invalid 'controls' list {controls}. Must be a list of valid qubit indices.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"_apply_mcx_kernel_sv: Invalid 'target' qubit {target}. Must be a valid qubit index.")
            raise ValueError(f"Invalid target qubit index {target}.")
        if target in controls:
            self._internal_logger.error(f"_apply_mcx_kernel_sv: Target qubit ({target}) cannot be in the controls list ({controls}).")
            raise ValueError("Target qubit for MCX gate cannot be in the controls list.")
            
        # 检查是否有重复的控制比特
        if len(set(controls)) != len(controls):
             self._internal_logger.error(f"_apply_mcx_kernel_sv: 'controls' list contains duplicate qubit indices: {controls}.")
             raise ValueError("Controls list for MCX gate must not contain duplicate qubit indices.")

        self._internal_logger.debug(f"Applying MCX kernel on controls={controls}, target={target}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用位掩码技术高效检查控制条件
        # 创建一个掩码，其中所有控制比特对应的位都为 1
        full_control_mask = sum(1 << q for q in controls)
        target_mask = 1 << target
        
        is_list_backend = isinstance(sv, list)

        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量，每次处理一对可能需要交换的振幅
        for i in range(dim):
            # 检查是否满足控制条件，并且为了避免重复交换，只处理目标比特为 0 的情况
            if (i & full_control_mask) == full_control_mask and (i & target_mask) == 0:
                # i 对应 |...1...1_controls...0_target...⟩ 基态
                # j 对应 |...1...1_controls...1_target...⟩ 基态
                j = i | target_mask
                
                # [健 robuste性改进] 对不同后端进行处理
                if is_list_backend:
                    # 对于纯Python列表，直接交换
                    sv[i], sv[j] = sv[j], sv[i]
                else: # CuPy 后端
                    # [核心修正] 使用 .item() 提取标量以安全地执行交换
                    temp = sv[i].item()
                    sv[i] = sv[j]
                    sv[j] = temp
                    
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 MCX 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()
    def _apply_mcp_kernel_sv(self, controls: List[int], target: int, angle: float):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用 Multi-Controlled-Phase (MCP) 门。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        MCP 门（也称为 n-CP Gate）是一个对角矩阵，其作用是：当且仅当所有控制比特 `controls`
        和目标比特 `target` 都处于 `|1⟩` 态时，给该计算基矢的振幅施加一个 `e^(i*angle)` 的相位。

        因此，此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`，
        检查其是否同时满足所有 `controls` 比特为 `1` 且 `target` 比特为 `1`。
        如果是，则将该索引对应的振幅 `sv[i]` 乘以 `e^(i*angle)`。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证，确保所有比特索引有效且不重叠。
        - 使用高效的位掩码技术进行条件检查。
        - 增加了详细的日志记录和注释。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            controls (List[int]): 控制量子比特的索引列表。
            target (int): 目标量子比特的索引。
            angle (float): 旋转角度 (单位：弧度)。

        Raises:
            ValueError: 如果 `controls` 或 `target` 包含无效或重叠的量子比特索引。
            TypeError: 如果 `controls` 不是一个列表，`target` 不是一个整数，或 `angle` 不是一个数值类型。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"_apply_mcp_kernel_sv: Invalid 'controls' list {controls}. Must be a list of valid qubit indices.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"_apply_mcp_kernel_sv: Invalid 'target' qubit {target}. Must be a valid qubit index.")
            raise ValueError(f"Invalid target qubit index {target}.")
        if target in controls:
            self._internal_logger.error(f"_apply_mcp_kernel_sv: Target qubit ({target}) cannot be in the controls list ({controls}).")
            raise ValueError("Target qubit for MCP gate cannot be in the controls list.")
        if not isinstance(angle, (float, int)):
            self._internal_logger.error(f"_apply_mcp_kernel_sv: 'angle' must be a numeric type, but got {type(angle).__name__}.")
            raise TypeError(f"Angle must be a numeric type.")
            
        # 检查是否有重复的控制比特
        if len(set(controls)) != len(controls):
             self._internal_logger.error(f"_apply_mcp_kernel_sv: 'controls' list contains duplicate qubit indices: {controls}.")
             raise ValueError("Controls list for MCP gate must not contain duplicate qubit indices.")

        self._internal_logger.debug(f"Applying MCP({angle:.4f}) kernel on controls={controls}, target={target}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 使用位掩码技术高效检查条件。
        # 创建一个掩码，其中所有控制比特和目标比特对应的位都为 1。
        full_mask = sum(1 << q for q in controls) | (1 << target)
        
        # 使用后端抽象的数学函数获取相位因子 e^(i*angle)
        phase = self._backend.cmath.exp(1j * angle)
        
        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量的所有分量
        for i in range(dim):
            # 检查当前索引 `i` 是否与 `full_mask` 完全匹配。
            # 这等效于检查 `i` 中所有控制位和目标位是否都为 1。
            if (i & full_mask) == full_mask:
                # 如果是，则将对应的振幅乘以相位因子
                sv[i] *= phase
                
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 MCP 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()

    def _apply_mcu_kernel_sv(self, controls: List[int], target: int, u_matrix: List[List[complex]], **kwargs: Any):
        """
        [内部优化内核][最终修正增强版] 在态矢量上高效地应用一个多控制酉门 (MCU)。

        此内核直接在态矢量上进行原地修改，避免了构建和应用全局矩阵的开销。
        MCU 门的作用是：当且仅当所有控制比特 `controls` 都处于 `|1⟩` 态时，
        在目标比特 `target` 上应用一个单比特酉矩阵 `u_matrix`。

        此内核的逻辑是遍历整个态矢量，对于每一个索引 `i`：
        1. 检查 `controls` 比特是否都为 `1`。
        2. 如果是，则找到与 `i` 对应的、目标比特翻转的索引 `j`。
        3. 在 `sv[i]` 和 `sv[j]` 这对振幅上应用 `u_matrix` 变换。

        此版本经过了全面的健 robuste性增强，包括：
        - 严格的输入验证，确保所有比特索引有效且不重叠，`u_matrix` 格式正确。
        - 使用高效的位掩码技术进行条件检查。
        - 增加了对 CuPy 后端的特别处理（使用 `.item()`），以确保原地修改的正确性。
        - 增加了详细的日志记录和注释。
        - 确保在操作后调用归一化，以维持数值稳定性。

        Args:
            controls (List[int]): 控制量子比特的索引列表。
            target (int): 目标量子比特的索引。
            u_matrix (List[List[complex]]): 要应用的 2x2 单比特酉矩阵。

        Raises:
            ValueError: 如果 `controls` 或 `target` 包含无效或重叠的量子比特索引，或 `u_matrix` 格式不正确。
            TypeError: 如果输入参数类型不正确。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"_apply_mcu_kernel_sv: Invalid 'controls' list {controls}. Must be a list of valid qubit indices.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"_apply_mcu_kernel_sv: Invalid 'target' qubit {target}. Must be a valid qubit index.")
            raise ValueError(f"Invalid target qubit index {target}.")
        if target in controls:
            self._internal_logger.error(f"_apply_mcu_kernel_sv: Target qubit ({target}) cannot be in the controls list ({controls}).")
            raise ValueError("Target qubit for MCU gate cannot be in the controls list.")
        if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
                isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
                isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2):
            self._internal_logger.error(f"_apply_mcu_kernel_sv: 'u_matrix' must be a 2x2 nested list of complex numbers. Got {type(u_matrix).__name__}.")
            raise ValueError("`u_matrix` for MCU must be a 2x2 nested list of complex numbers.")
            
        # 检查是否有重复的控制比特
        if len(set(controls)) != len(controls):
             self._internal_logger.error(f"_apply_mcu_kernel_sv: 'controls' list contains duplicate qubit indices: {controls}.")
             raise ValueError("Controls list for MCU gate must not contain duplicate qubit indices.")

        self._internal_logger.debug(f"Applying MCU kernel on controls={controls}, target={target}.")

        # --- 步骤 2: 初始化和获取所需变量 ---
        dim = 1 << self.num_qubits
        sv = self._state_vector
        
        # 从 u_matrix 中提取元素
        u00, u01 = u_matrix[0]
        u10, u11 = u_matrix[1]
        
        # 使用位掩码技术高效检查控制条件
        full_control_mask = sum(1 << q for q in controls)
        target_mask = 1 << target
        
        is_list_backend = isinstance(sv, list)

        # --- 步骤 3: 核心计算逻辑 ---
        # 遍历态矢量，每次处理一对可能需要变换的振幅
        for i in range(dim):
            # 检查是否满足控制条件，并且为了避免重复计算，只处理目标比特为 0 的情况
            if (i & full_control_mask) == full_control_mask and (i & target_mask) == 0:
                idx0 = i # 对应 |...1...1_controls...0_target...⟩ 基态
                idx1 = i | target_mask # 对应 |...1...1_controls...1_target...⟩ 基态
                
                # [健 robuste性改进] 对不同后端进行处理
                if is_list_backend:
                    # 对于纯Python列表，直接访问
                    amp0 = sv[idx0]
                    amp1 = sv[idx1]
                else: # CuPy 后端
                    # [核心修正] 使用 .item() 提取标量以安全地执行计算
                    amp0 = sv[idx0].item()
                    amp1 = sv[idx1].item()
                
                # 根据 u_matrix 的定义计算新的振幅
                new_amp0 = u00 * amp0 + u01 * amp1
                new_amp1 = u10 * amp0 + u11 * amp1
                
                # 将计算出的新振幅写回态矢量
                if is_list_backend:
                    sv[idx0] = new_amp0
                    sv[idx1] = new_amp1
                else:  # CuPy 后端
                    sv[idx0] = new_amp0
                    sv[idx1] = new_amp1
                    
        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 MCU 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self._normalize()
    
    
    def _apply_apply_unitary_kernel_sv(self, unitary_data: Union[Any, 'TranspiledUnitary'], target_qubits: List[int]):
        """
        [内部优化内核][最终修正增强版] 应用由优化器生成的 `apply_unitary` 指令。

        此内核是 `nexus_optimizer` 优化流程的核心执行器。它能够处理两种输入格式：
        1.  `TranspiledUnitary` 对象：这是推荐的、包含丰富元数据的新格式。
            内核会从中提取实际的 `unitary_matrix` 用于计算，并可以利用
            其他元数据进行日志记录或调试。
        2.  裸矩阵 (Any): 为了保持向后兼容性和灵活性，内核也能处理直接
            传递的酉矩阵（Python列表或后端数组）。

        工作流程:
        1.  **输入解析**: 检查 `unitary_data` 的类型。如果是 `TranspiledUnitary`，
            则提取其内部的 `unitary_matrix`；否则，直接使用 `unitary_data`。
        2.  **验证**: 对提取出的矩阵和 `target_qubits` 列表进行全面的类型和
            维度验证。
        3.  **全局算子构建**:
            - 如果酉矩阵作用于系统的所有量子比特，则它本身就是全局算子。
            - 否则，调用 `_build_global_operator_multi_qubit` 方法，将局部
              算子“提升”为作用于整个系统的大型全局算子。
        4.  **应用**: 调用 `_apply_global_unitary` 方法，将构建好的全局算子
            高效地应用到当前的状态上（无论是态矢量还是矩阵累积模式）。

        Args:
            unitary_data (Union[Any, 'TranspiledUnitary']): 
                要应用的酉矩阵数据。可以是 `TranspiledUnitary` 对象或裸矩阵。
            target_qubits (List[int]): 
                酉矩阵作用的全局量子比特索引列表。

        Raises:
            TypeError: 如果输入参数的类型不正确。
            ValueError: 如果 `target_qubits` 无效，或矩阵维度与 `target_qubits`
                        的数量不匹配。
            RuntimeError: 如果在构建或应用全局算子时发生内部错误。
        """
        log_prefix = f"[_StateVectorEntity._apply_apply_unitary_kernel_sv(N={self.num_qubits})]"
        
        # --- 步骤 1: 解析输入，从 TranspiledUnitary 中提取裸矩阵 ---
        local_unitary_op: Any
        source_description = "direct matrix" # 默认为直接传递的矩阵
        
        # --- [核心修复：将 isinstance 替换为鸭子类型检查] ---
        # 检查 unitary_data 是否具有 TranspiledUnitary 的关键属性，而不是检查其具体类型。
        # 这可以避免因复杂的跨模块导入导致 `isinstance` 失效的问题。
        if (hasattr(unitary_data, 'unitary_matrix') and 
            hasattr(unitary_data, 'num_qubits') and 
            hasattr(unitary_data, 'source_circuit_hash')):
            local_unitary_op = unitary_data.unitary_matrix
            # 从元数据中获取更丰富的日志信息
            source_description = f"TranspiledUnitary from hash '{unitary_data.source_circuit_hash[:8] if unitary_data.source_circuit_hash else 'N/A'}'"
            self._internal_logger.debug(f"{log_prefix} Applying {source_description} on qubits {target_qubits}.")
        else:
            # 兼容直接传递裸矩阵的旧方式
            local_unitary_op = unitary_data
            self._internal_logger.debug(f"{log_prefix} Applying {source_description} on qubits {target_qubits}.")

        # --- 步骤 2: 对提取出的矩阵和目标比特进行严格验证 ---
        
        # a) 验证 target_qubits
        if not isinstance(target_qubits, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in target_qubits):
            self._internal_logger.error(f"{log_prefix} Invalid 'target_qubits' list {target_qubits}.")
            raise ValueError("'target_qubits' must be a list of valid qubit indices.")
        
        # b) 验证 local_unitary_op 的类型和维度
        is_cupy_backend = isinstance(self._backend, CuPyBackendWrapper)
        expected_array_type = self._backend._cp.ndarray if is_cupy_backend and self._backend._cp is not None else type(None)
        
        if not isinstance(local_unitary_op, (list, expected_array_type)):
            self._internal_logger.error(f"{log_prefix} The extracted 'unitary_matrix' must be a list or a compatible backend array, but got {type(local_unitary_op).__name__}.")
            raise TypeError("The extracted 'unitary_matrix' must be a list or a compatible backend array.")
            
        op_dim_check = 1 << len(target_qubits)
        op_shape_check = local_unitary_op.shape if hasattr(local_unitary_op, 'shape') else (len(local_unitary_op), len(local_unitary_op[0]) if local_unitary_op else 0)
        
        if op_shape_check != (op_dim_check, op_dim_check):
            self._internal_logger.error(
                f"{log_prefix} Dimension mismatch. The provided unitary matrix has shape {op_shape_check}, "
                f"but it should be ({op_dim_check}, {op_dim_check}) for the {len(target_qubits)} target qubits."
            )
            raise ValueError(
                f"The provided unitary matrix has shape {op_shape_check}, which is inconsistent with the "
                f"{len(target_qubits)} target qubits specified."
            )

        # --- 步骤 3: 构建全局算子 ---
        
        # 将传入的 unitary_matrix 转换为后端兼容的类型
        backend_compatible_unitary = self._backend._ensure_cupy_array(local_unitary_op) if is_cupy_backend else local_unitary_op
        
        # 检查是否作用于所有比特
        if len(target_qubits) == self.num_qubits and sorted(target_qubits) == list(range(self.num_qubits)):
            global_op = backend_compatible_unitary
        else:
            # 如果不是作用于所有比特，需要构建全局算子
            global_op = self._build_global_operator_multi_qubit(target_qubits, backend_compatible_unitary)
        
        # --- 步骤 4: 应用全局算子 ---
        self._apply_global_unitary(global_op)

    # --- 核心辅助方法: 构建全局算子 (用于非优化门的回退路径) ---
    # --- 文件: quantum_core.py ---
    # (位于 _StateVectorEntity 类内部)

    def _build_global_operator_multi_qubit(self, target_qubits: List[int], local_operator: Any) -> Any:
        """
        [健壮性与索引逻辑最终修复版 - 2025-10-28 v6 - ABSOLUTELY FINAL]
        为作用于任意多个（可能非连续）量子比特的多比特门构建全局算子。
        此版本彻底重写了索引映射逻辑，以确保与内核和标准张量积的比特序约定一致。

        此函数的核心任务是将一个小的、作用于局部子系统（由 `target_qubits` 定义）
        的酉矩阵 `local_operator`，“提升”为一个作用于整个系统的大型全局酉矩阵。
        
        工作流程:
        1. 初始化一个与整个系统维度相同的全零全局矩阵。
        2. 遍历这个大型全局矩阵的每一个元素 (global_row, global_col)。
        3. 对于每个元素，首先检查它是否可能为非零。一个元素只有在作用于
           “旁观者”量子比特（不在 `target_qubits` 中）的部分是单位矩阵时才可能非零。
           这意味着对于所有旁观者比特 q_other, global_row 和 global_col 在
           q_other 上的比特值必须相同。
        4. 如果满足上述条件，则将 global_row 和 global_col 在 `target_qubits`
           上的比特值提取出来，并根据 `target_qubits` 列表的顺序将它们
           重新组合成局部的行索引 `local_row_idx` 和列索引 `local_col_idx`。
           这个重组过程是修复的核心，它确保了正确的比特序映射。
        5. 使用这两个局部索引从 `local_operator` 中查找对应的值，并将其赋给
           全局矩阵的 (global_row, global_col) 位置。

        Args:
            target_qubits (List[int]):
                局部算子作用的目标量子比特的全局索引列表。列表的顺序至关重要，
                它定义了局部算子矩阵基矢的顺序。例如，对于 CNOT(c,t)，
                `target_qubits` 应该是 `[c, t]`。
            local_operator (Any):
                作用于 `target_qubits` 上的局部酉矩阵（Python列表或后端数组）。

        Returns:
            Any:
                一个完整的、作用于整个系统的全局酉矩阵，其类型与当前后端兼容。

        Raises:
            ValueError: 如果 `target_qubits` 无效或 `local_operator` 维度不匹配。
        """
        log_prefix = f"[_StateVectorEntity._build_global_operator_multi_qubit(N={self.num_qubits})]"
        
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(target_qubits, list) or not target_qubits:
            raise ValueError("'target_qubits' must be a non-empty list of integers.")
        
        num_local_qubits = len(target_qubits)
        if len(set(target_qubits)) != num_local_qubits:
            raise ValueError("target_qubits list must contain unique qubit indices.")
        
        for q in target_qubits:
            if not (0 <= q < self.num_qubits):
                raise ValueError(f"Target qubit index {q} is out of range.")
        
        local_dim = 1 << num_local_qubits
        op_shape = local_operator.shape if hasattr(local_operator, 'shape') else (len(local_operator), len(local_operator[0]) if local_operator else (0,0))
        
        if op_shape != (local_dim, local_dim):
            raise ValueError(f"Shape mismatch: local operator {op_shape} vs expected {(local_dim, local_dim)}.")

        # --- 步骤 2: 初始化和准备 ---
        total_dim = 1 << self.num_qubits
        global_op = self._backend.zeros((total_dim, total_dim), dtype=complex)
        
        other_qubits = [q for q in range(self.num_qubits) if q not in target_qubits]
        
        is_list_backend = isinstance(global_op, list)
        is_local_op_list = isinstance(local_operator, list)
        
        # --- 步骤 3: [最终核心修复] 遍历全局矩阵的每个元素，确定其值 ---
        for global_row in range(total_dim):
            for global_col in range(total_dim):
                
                # a) 检查旁观者比特是否匹配。如果不匹配，则该矩阵元素必为0。
                is_identity_on_others = True
                for q_other in other_qubits:
                    mask = 1 << q_other
                    if (global_row & mask) != (global_col & mask):
                        is_identity_on_others = False
                        break
                
                if not is_identity_on_others:
                    continue

                # b) [最终逻辑] 将全局索引反向映射到局部索引，同时修正比特序。
                # 局部算子的矩阵约定是基于 target_qubits 列表的顺序。
                # 例如，如果 target_qubits = [c, t]，则局部算子的基矢顺序是 |ct⟩。
                # 列表中的第一个元素 (c) 对应局部空间的最高位比特。
                local_row_idx = 0
                local_col_idx = 0
                for i, q_target in enumerate(target_qubits):
                    # i=0 对应 target_qubits 的第一个元素，它应该是局部算子中的最高位比特。
                    local_bit_weight = num_local_qubits - 1 - i
                    
                    # 检查该全局比特在全局索引中的值，并将其贡献加到局部索引中。
                    if (global_row >> q_target) & 1:
                        local_row_idx |= (1 << local_bit_weight)
                    if (global_col >> q_target) & 1:
                        local_col_idx |= (1 << local_bit_weight)

                # c) 从局部算子中获取值并赋值给全局矩阵的相应位置。
                if is_local_op_list:
                    value_to_assign = local_operator[local_row_idx][local_col_idx]
                else: # CuPy/NumPy array
                    value_to_assign = local_operator[local_row_idx, local_col_idx]
                
                # 优化：只有当值非零时才进行赋值操作。
                if not self._backend.isclose(complex(value_to_assign), 0.0 + 0.0j, atol=1e-12):
                    if is_list_backend:
                        global_op[global_row][global_col] = value_to_assign
                    else: # CuPy/NumPy array
                        global_op[global_row, global_col] = value_to_assign

        return global_op
    
    
    def _build_generic_unitary_operator(self, macro_name: str, args: List[Any]) -> Any:
        """
        [健-壮性改进版] [v1.5.7 unhashable type 'list' 修复版]
        根据宏名称和参数，自动构建其对应的全局酉矩阵。
        此版本修复了当宏的参数包含列表（如 apply_unitary 的矩阵）时，
        由于 lru_cache 无法处理不可哈希类型而导致的 TypeError。

        修复逻辑:
        1. 对 'apply_unitary' 进行特殊处理，直接提取其矩阵参数，绕过宏查找。
        2. 对所有其他宏，在调用缓存的 _get_macro_unitary 之前，
           将参数中的所有列表递归地转换为元组，使其变为可哈希的。

        Args:
            macro_name (str): 宏的名称。
            args (List[Any]): 宏的参数。

        Returns:
            Any: 宏的全局酉矩阵。

        Raises:
            ValueError: 如果宏未注册或无法构建算子。
            TypeError: 如果参数类型不正确。
        """
        log_prefix = f"[QuantumState._build_generic_unitary_operator(N={self.num_qubits}, Macro='{macro_name}')]"

        try:
            # 1. 获取宏的局部酉矩阵
            # 导入 AlgorithmBuilders
            try:
                from . import AlgorithmBuilders
            except ImportError:
                AlgorithmBuilders = globals().get('AlgorithmBuilders')
                if AlgorithmBuilders is None:
                    raise RuntimeError("AlgorithmBuilders class is not available in current scope. Ensure quantum_core is properly imported.")

            # --- [核心修复] ---
            local_unitary: Any
            if macro_name == 'apply_unitary':
                # 路径 A: 特殊处理 apply_unitary。其局部酉矩阵就是它的第一个参数。
                if not args or not (isinstance(args[0], list) or hasattr(args[0], 'shape')):
                    self._internal_logger.error(f"[{log_prefix}] 'apply_unitary' instruction is missing its matrix argument.")
                    raise ValueError("'apply_unitary' instruction is missing its matrix argument.")
                local_unitary = args[0]
            else:
                # 路径 B: 对于所有其他宏，净化参数使其可哈希，然后调用缓存函数。
                def _make_hashable(arg: Any) -> Any:
                    """递归地将列表转换为元组。"""
                    if isinstance(arg, list):
                        return tuple(_make_hashable(e) for e in arg)
                    # 在未来可以扩展以处理其他可变类型，如 dict -> frozenset
                    return arg

                # 将所有参数（包括嵌套的）转换为可哈希类型
                sanitized_args_for_cache = tuple(_make_hashable(arg) for arg in args)
                
                # 现在可以安全地调用被 lru_cache 装饰的函数
                local_unitary = AlgorithmBuilders._get_macro_unitary(macro_name, sanitized_args_for_cache)
            # --- [修复结束] ---
            
            # 2. 从参数中提取涉及的量子比特
            macro_info = AlgorithmBuilders.get_macro_definition(macro_name)
            
            # [修正] 对 apply_unitary 也需要获取其比特数信息
            num_local_qubits_from_macro: int
            if macro_name == 'apply_unitary':
                # apply_unitary 没有宏定义，我们需要从其参数中推断
                # op_args for apply_unitary is [matrix, target_qubits]
                if len(args) < 2 or not isinstance(args[1], list):
                     raise ValueError("'apply_unitary' is missing its target_qubits list argument.")
                num_local_qubits_from_macro = len(args[1])
            elif macro_info is None:
                self._internal_logger.error(f"[{log_prefix}] Macro '{macro_name}' not found for operator building.")
                raise ValueError(f"Macro '{macro_name}' not found for operator building.")
            else:
                _, num_local_qubits_from_macro, _ = macro_info
            
            target_qubits: List[int]

            # [修正] 确保 apply_unitary 也能正确提取 target_qubits
            if macro_name == 'apply_unitary':
                target_qubits = args[1] # 第二个参数就是 target_qubits
            elif num_local_qubits_from_macro == 0: # 可变比特数宏 (MCX, MCZ, MCP, MCU)
                if self._backend.builtins.len(args) < 2:
                    self._internal_logger.error(f"[{log_prefix}] Insufficient arguments for variable-qubit macro '{macro_name}'. Expected controls (list) and target (int). Got {args}.")
                    raise ValueError(f"Insufficient arguments for variable-qubit macro '{macro_name}'.")
                
                controls_from_args = args[0]
                target_from_args = args[1]
                
                if not isinstance(controls_from_args, list) or not self._backend.builtins.all(isinstance(q, int) for q in controls_from_args):
                     raise TypeError(f"[{log_prefix}] Controls for variable-qubit macro '{macro_name}' must be a list of integers, got {type(controls_from_args).__name__}.")
                if not isinstance(target_from_args, int):
                     raise TypeError(f"[{log_prefix}] Target for variable-qubit macro '{macro_name}' must be an integer, got {type(target_from_args).__name__}.")

                # 对于构建全局算子，target_qubits 的顺序很重要，它定义了局部空间的基矢顺序
                # 我们使用 [controls..., target] 的顺序来匹配 local_operator 的构建方式
                target_qubits = controls_from_args + [target_from_args]
                
            else: # 固定比特数宏 (例如 H, CNOT, Toffoli)
                qubit_indices_in_args = [arg for arg in args if self._backend.builtins.isinstance(arg, self._backend.builtins.int) and 0 <= arg < self.num_qubits]
                if self._backend.builtins.len(self._backend.builtins.set(qubit_indices_in_args)) != num_local_qubits_from_macro:
                    self._internal_logger.warning(
                        f"[{log_prefix}] Number of distinct qubit arguments ({self._backend.builtins.len(self._backend.builtins.set(qubit_indices_in_args))}) "
                        f"does not match macro definition ({num_local_qubits_from_macro}). "
                        "Attempting to proceed with available qubit arguments. This might lead to incorrect mapping."
                    )
                target_qubits = qubit_indices_in_args
            
            # 3. 将局部矩阵提升为全局算子
            global_op = self._build_global_operator_multi_qubit(target_qubits, local_unitary)
            return global_op
            
        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] Failed to build global operator for macro: {e}", exc_info=True)
            raise RuntimeError(f"Failed to build global operator for '{macro_name}': {e}") from e
    # --- 核心方法: 归一化态矢量 (仅限1D态矢量) ---
    def _normalize(self):
        """
        [健壮性与模式感知增强版] 对当前态矢量进行归一化，使其总概率为1 (范数为1)。

        此方法仅在实体处于“态矢量模式”（即 `_state_vector` 是一维向量）时执行。
        它通过计算态矢量的范数 `||ψ||`，然后将每个振幅除以该范数来实现归一化：
        `|ψ_norm⟩ = |ψ⟩ / ||ψ||`。

        核心特性:
        - **模式感知**: 自动检测 `_state_vector` 的维度。如果是二维矩阵（矩阵累积模式），
          则会安全地跳过归一化，因为对一个酉矩阵进行这种归一化是没有意义的。
        - **性能优化**: 只有当态矢量的范数平方与 1 的偏差超过了设定的容差 (1e-9) 时，
          才会执行实际的（可能是昂贵的）除法操作。
        - **后端无关**: 所有数值计算都通过 `self._backend` 的抽象接口进行，
          确保在 PurePython 和 CuPy 后端下都能正确工作。
        - **健壮性**: 增加了对范数为零的检查，以防止因除以零而崩溃。

        Raises:
            RuntimeError: 如果在计算或归一化过程中发生任何未预期的底层错误。
        """
        log_prefix = "[_StateVectorEntity._normalize]"
        
        # --- 步骤 1: 检查状态向量是否已初始化 ---
        if self._state_vector is None:
            self._internal_logger.warning(f"{log_prefix} Called on a None state vector. Skipping normalization.")
            return

        # --- 步骤 2: 模式感知 - 检查是否为矩阵模式 ---
        # 通过检查维度来判断。如果是二维，则为矩阵模式。
        is_matrix_mode = (hasattr(self._state_vector, 'ndim') and self._state_vector.ndim == 2) or \
                         (isinstance(self._state_vector, list) and self._state_vector and isinstance(self._state_vector[0], list))

        if is_matrix_mode:
            self._internal_logger.debug(f"{log_prefix} Skipping normalization because the entity is in matrix accumulation mode.")
            return

        # --- 步骤 3: 计算范数平方并进行有效性检查 ---
        try:
            # 调用后端提供的 `norm_sq` 方法，该方法计算 Σ|amp_i|^2
            norm_sq_float = self._backend.norm_sq(self._state_vector)

            # 检查范数是否接近于零。如果是，则无法归一化。
            if self._backend.isclose(norm_sq_float, 0.0, atol=1e-12):
                self._internal_logger.critical(
                    f"{log_prefix} Cannot normalize state vector because its norm is close to zero ({norm_sq_float:.2e}). "
                    "The quantum state may be corrupted. Normalization skipped."
                )
                return

            # --- 步骤 4: 仅在需要时执行归一化 ---
            # 如果范数平方已经足够接近 1，则无需进行昂贵的除法操作。
            if not self._backend.isclose(norm_sq_float, 1.0, atol=1e-9):
                # 使用后端计算平方根
                norm = self._backend.sqrt(norm_sq_float)
                self._internal_logger.debug(f"{log_prefix} Normalizing state vector with norm {norm:.8f} (norm_sq={norm_sq_float:.8f})")
                
                # 执行除法操作。
                # 无论是 PurePython (list) 还是 CuPy (ndarray)，
                # 都可以通过 `/=` 或 `/` 操作符进行除法。
                if isinstance(self._state_vector, list):
                    # PurePythonBackend: 需要手动遍历
                    self._state_vector = [amp / norm for amp in self._state_vector]
                else:
                    # CuPyBackendWrapper: 支持原地除法
                    self._state_vector /= norm
            else:
                self._internal_logger.debug(f"{log_prefix} State vector is already normalized (norm_sq={norm_sq_float:.8f}, within tolerance).")

        except Exception as e:
            self._internal_logger.error(f"{log_prefix} An error occurred during state vector normalization: {e}", exc_info=True)
            # 将底层错误包装成一个更通用的 RuntimeError
            raise RuntimeError("State vector normalization failed.") from e
    def _apply_global_unitary(self, global_unitary: Any):
        """
        [新增修复方法] 将一个全局酉算子应用于内部的 1D 态矢量或 2D 矩阵。

        此方法是 `run_circuit_on_entity` 通用后备路径的核心执行器。它根据当前实体
        所处的模式（通过检查 `_state_vector` 的维度来判断），执行正确的线性代数操作。

        - **态矢量模式 (`_state_vector` is 1D):**
          计算 `|ψ'⟩ = U |ψ⟩`，其中 U 是 `global_unitary`。
        - **矩阵累积模式 (`_state_vector` is 2D):**
          计算 `U_total' = U @ U_total`，其中 U 是 `global_unitary`，
          `U_total` 是已累积的酉矩阵。

        此方法完全依赖于后端抽象（`self._backend.dot`），使其能够同时在
        PurePython 和 CuPy 后端下正确工作。

        Args:
            global_unitary (Any): 全局酉算子矩阵（应为 Python 列表或 CuPy/NumPy 数组）。

        Raises:
            RuntimeError: 如果状态未初始化或底层计算失败。
            TypeError: 如果 `global_unitary` 类型不正确。
            ValueError: 如果 `global_unitary` 的维度与状态维度不匹配。
        """
        log_prefix = f"[_StateVectorEntity._apply_global_unitary(N={self.num_qubits})]"
        
        # 检查状态是否已初始化
        if self._state_vector is None:
            self._internal_logger.error(f"{log_prefix} Cannot apply unitary: state vector has not been initialized.")
            raise RuntimeError("Cannot apply unitary: state vector has not been initialized.")
        
        # --- 步骤 1: 输入验证和后端兼容性检查 ---
        is_cupy_backend = isinstance(self._backend, CuPyBackendWrapper)
        
        # 根据后端确定可接受的数组类型
        expected_array_type = type(None)
        if is_cupy_backend and self._backend._cp is not None:
            expected_array_type = self._backend._cp.ndarray

        # 验证 global_unitary 的类型
        if not isinstance(global_unitary, (list, expected_array_type)):
             self._internal_logger.error(f"{log_prefix} 'global_unitary' must be a list or a compatible backend array, but got {type(global_unitary).__name__}.")
             raise TypeError("'global_unitary' must be a list or a compatible backend array.")

        # 检查维度是否匹配
        op_dim = len(global_unitary) if isinstance(global_unitary, list) else global_unitary.shape[0]
        state_dim = 1 << self.num_qubits
        if op_dim != state_dim:
            self._internal_logger.error(f"{log_prefix} Dimension mismatch. Global unitary has dim {op_dim}, but state has dim {state_dim}.")
            raise ValueError(f"Dimension mismatch between global unitary ({op_dim}) and state ({state_dim}).")

        # --- 步骤 2: 判断模式并执行相应的点积操作 ---
        is_matrix_mode = (hasattr(self._state_vector, 'ndim') and self._state_vector.ndim == 2) or \
                        (isinstance(self._state_vector, list) and self._state_vector and isinstance(self._state_vector[0], list))
        
        try:
            if is_matrix_mode:
                self._internal_logger.debug(f"{log_prefix} Applying unitary in matrix accumulation mode (U_new = U_op @ U_old).")
                # 计算 U_new = U_op @ U_old
                self._state_vector = self._backend.dot(global_unitary, self._state_vector)
            else: # 态矢量模式
                self._internal_logger.debug(f"{log_prefix} Applying unitary to 1D state vector (|ψ'> = U|ψ>).")
                
                # 计算 |ψ'> = U|ψ>。self._backend.dot 能正确处理矩阵-向量乘法。
                result_vec = self._backend.dot(global_unitary, self._state_vector)
                
                # 后端应该返回一个1D向量。如果返回了(N,1)的2D矩阵，我们需要将其扁平化。
                if isinstance(result_vec, list) and result_vec and isinstance(result_vec[0], list):
                    # PurePythonBackend 的 dot 矩阵-向量乘法返回 1D list，所以这段逻辑主要用于防御
                    if len(result_vec[0]) == 1: # 检查是否是 (N, 1) 形状
                        self._state_vector = [row[0] for row in result_vec]
                    else:
                        raise RuntimeError(f"Unexpected 2D matrix shape {len(result_vec)}x{len(result_vec[0])} from dot product in statevector mode.")
                elif hasattr(result_vec, 'ndim') and result_vec.ndim == 2:
                    if result_vec.shape[1] == 1: # 检查是否是 (N, 1) 形状
                        self._state_vector = result_vec.ravel() # 使用 ravel() 高效扁平化
                    else:
                        raise RuntimeError(f"Unexpected 2D matrix shape {result_vec.shape} from dot product in statevector mode.")
                else: # 如果结果已经是 1D 向量
                    self._state_vector = result_vec
            
            # --- 步骤 3: 归一化 (仅在态矢量模式下) ---
            # 矩阵累积模式下不需要归一化。
            if not is_matrix_mode:
                self._normalize()

        except Exception as e:
            self._internal_logger.error(f"{log_prefix} An error occurred during state evolution via global unitary: {e}", exc_info=True)
            raise RuntimeError("Numerical error during state evolution via global unitary.") from e
    
    def _build_mcz_operator(self, controls: List[int], target: int) -> Any:
        """
        [健壮性与后端优化增强版] 构建 Multi-Controlled-Z (MCZ) 门的全局酉矩阵。

        此方法用于 `run_circuit_on_entity` 的后备路径，当无法使用优化内核时
        （例如，在矩阵累积模式下），需要构建完整的算子矩阵。

        MCZ 门是一个对角矩阵，其作用是：当且仅当所有控制比特 `controls` 和目标比特 `target`
        都处于 `|1⟩` 态时，给该计算基矢的振幅施加一个 `-1` 的相位。在矩阵表示中，
        这意味着在对应的对角线位置上，元素值从 1 变为 -1。

        核心增强功能:
        - 增加了对 `controls` 列表和 `target` 的全面验证，包括范围、类型、唯一性检查。
        - 实现了后端感知的优化：对于 CuPy/NumPy 后端，使用高效的向量化和布尔索引
          来批量修改对角线元素；对于 PurePython 后端，使用清晰的 for 循环。
        - 增加了详尽的文档和内部注释。

        Args:
            controls (List[int]): 控制量子比特的索引列表。
            target (int): 目标量子比特的索引。

        Returns:
            Any: MCZ 门的全局酉矩阵，其类型与当前后端匹配。

        Raises:
            ValueError: 如果 `controls` 或 `target` 包含无效或重叠的量子比特索引。
            TypeError: 如果 `controls` 不是一个列表或 `target` 不是一个整数。
        """
        log_prefix = f"[_StateVectorEntity._build_mcz_operator(N={self.num_qubits})]"

        # --- 步骤 1: 极度严格的输入验证 ---
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"{log_prefix} Invalid 'controls' list {controls}. Must be a list of valid qubit indices.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"{log_prefix} Invalid 'target' qubit {target}. Must be a valid qubit index.")
            raise ValueError(f"Invalid target qubit index {target}.")
        
        if target in controls:
            self._internal_logger.error(f"{log_prefix} Target qubit ({target}) cannot be in the controls list ({controls}).")
            raise ValueError("Target qubit for MCZ gate cannot be in the controls list.")
        
        if len(set(controls)) != len(controls):
             self._internal_logger.error(f"{log_prefix} 'controls' list contains duplicate qubit indices: {controls}.")
             raise ValueError("Controls list for MCZ gate must not contain duplicate qubit indices.")

        self._internal_logger.debug(f"{log_prefix} Building MCZ operator matrix for controls={controls}, target={target}.")

        # --- 步骤 2: 初始化和准备 ---
        op_dim = 1 << self.num_qubits
        # 创建一个单位矩阵作为基础
        mcz_global_u = self._backend.eye(op_dim, dtype=complex)
        
        # 使用位掩码技术高效检查条件。
        # 创建一个掩码，其中所有控制比特和目标比特对应的位都为 1。
        full_mask = sum(1 << q for q in controls) | (1 << target)
        
        # --- 步骤 3: 根据后端类型构建矩阵 ---
        if isinstance(mcz_global_u, list):
            # --- 路径 A: PurePythonBackend ---
            # 遍历所有对角线元素
            for i in range(op_dim):
                # 检查当前索引 `i` 是否与 `full_mask` 完全匹配。
                # 这等效于检查 `i` 中所有控制位和目标位是否都为 1。
                if (i & full_mask) == full_mask:
                    # 如果匹配，将该对角线元素乘以 -1
                    mcz_global_u[i][i] *= -1.0
        else:
            # --- 路径 B: CuPy/NumPy Backend (使用向量化操作) ---
            # 1. 创建一个从 0 到 op_dim-1 的索引数组
            indices = self._backend.arange(op_dim)
            
            # 2. 创建一个布尔掩码，标记所有满足条件的对角线位置
            # (indices & full_mask) == full_mask 会对每个索引进行位检查
            target_indices_mask = (indices & full_mask) == full_mask
            
            # 3. 使用高级布尔索引来一次性地修改所有目标对角线元素
            mcz_global_u[target_indices_mask, target_indices_mask] *= -1.0
                
        return mcz_global_u
    def _build_mcx_operator(self, controls: List[int], target: int) -> Any:
        """
        [健壮性与后端优化增强版] 构建 Multi-Controlled-X (MCX) 门的全局酉矩阵。

        此方法用于 `run_circuit_on_entity` 的后备路径，当无法使用优化内核时
        （例如，在矩阵累积模式下），需要构建完整的算子矩阵。

        MCX 门（也称 n-Toffoli 门）的作用是：当且仅当所有控制比特 `controls` 都处于 `|1⟩` 态时，
        在目标比特 `target` 上应用一个 Pauli-X 门（翻转）。在矩阵表示中，这相当于
        在一个单位矩阵的基础上，交换满足控制条件的 |...1...0...⟩ 和 |...1...1...⟩
        基态所对应的行和列。

        核心增强功能:
        - 增加了对 `controls` 列表和 `target` 的全面验证，包括范围、类型、唯一性检查。
        - 实现了后端感知的优化：对于 CuPy/NumPy 后端，使用高效的向量化和布尔索引
          来批量地执行交换操作；对于 PurePython 后端，使用清晰的 for 循环。
        - 增加了详尽的文档和内部注释。

        Args:
            controls (List[int]): 控制量子比特的索引列表。
            target (int): 目标量子比特的索引。

        Returns:
            Any: MCX 门的全局酉矩阵，其类型与当前后端匹配。

        Raises:
            ValueError: 如果 `controls` 或 `target` 包含无效或重叠的量子比特索引。
            TypeError: 如果 `controls` 不是一个列表或 `target` 不是一个整数。
        """
        log_prefix = f"[_StateVectorEntity._build_mcx_operator(N={self.num_qubits})]"

        # --- 步骤 1: 极度严格的输入验证 ---
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"{log_prefix} Invalid 'controls' list {controls}. Must be a list of valid qubit indices.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"{log_prefix} Invalid 'target' qubit {target}. Must be a valid qubit index.")
            raise ValueError(f"Invalid target qubit index {target}.")
        
        if target in controls:
            self._internal_logger.error(f"{log_prefix} Target qubit ({target}) cannot be in the controls list ({controls}).")
            raise ValueError("Target qubit for MCX gate cannot be in the controls list.")
        
        if len(set(controls)) != len(controls):
             self._internal_logger.error(f"{log_prefix} 'controls' list contains duplicate qubit indices: {controls}.")
             raise ValueError("Controls list for MCX gate must not contain duplicate qubit indices.")

        self._internal_logger.debug(f"{log_prefix} Building MCX operator matrix for controls={controls}, target={target}.")

        # --- 步骤 2: 初始化和准备 ---
        op_dim = 1 << self.num_qubits
        # 创建一个单位矩阵作为基础
        mcx_global_u = self._backend.eye(op_dim, dtype=complex)
        
        # 创建控制位掩码和目标位掩码
        control_mask = sum(1 << q for q in controls)
        target_mask = 1 << target

        # --- 步骤 3: 根据后端类型构建矩阵 ---
        if isinstance(mcx_global_u, list):
            # --- 路径 A: PurePythonBackend ---
            # 遍历所有可能的基态索引
            for i in range(op_dim):
                # 检查是否满足所有控制条件，并且为了避免重复操作，只处理目标比特为 0 的情况
                if (i & control_mask) == control_mask and (i & target_mask) == 0:
                    # i 是 |...1...1_controls...0_target...⟩
                    # j 是 |...1...1_controls...1_target...⟩
                    j = i | target_mask
                    
                    # 在单位矩阵中执行交换操作
                    # |i⟩<i| 和 |j⟩<j| 变为 |i⟩<j| 和 |j⟩<i|
                    mcx_global_u[i][i] = 0.0 + 0.0j
                    mcx_global_u[j][j] = 0.0 + 0.0j
                    mcx_global_u[i][j] = 1.0 + 0.0j
                    mcx_global_u[j][i] = 1.0 + 0.0j
        else:
            # --- 路径 B: CuPy/NumPy Backend (使用向量化操作) ---
            # 1. 创建一个从 0 到 op_dim-1 的索引数组
            indices = self._backend.arange(op_dim)
            
            # 2. 找到所有满足“控制位为1”且“目标位为0”的基态索引
            base_indices = indices[((indices & control_mask) == control_mask) & ((indices & target_mask) == 0)]
            
            # 3. 只有在找到了这样的基态时才执行操作
            if hasattr(base_indices, 'size') and base_indices.size > 0:
                # 计算与它们配对的、目标位为1的索引
                swapped_indices = base_indices | target_mask
                
                # 4. 使用高级索引批量地执行交换操作
                # 将 |i⟩<i| 和 |j⟩<j| 的对角线元素清零
                mcx_global_u[base_indices, base_indices] = 0
                mcx_global_u[swapped_indices, swapped_indices] = 0
                
                # 将 |i⟩<j| 和 |j⟩<i| 的非对角线元素置为1
                mcx_global_u[base_indices, swapped_indices] = 1
                mcx_global_u[swapped_indices, base_indices] = 1
                
        return mcx_global_u
    
    def _build_local_mcx_operator(self, num_qubits: int, controls: List[int], target: int) -> Any:
        """
        [内部辅助方法][新增] 构建一个局部的 Multi-Controlled-X (MCX) 酉矩阵。
        此函数与 _build_mcx_operator 不同，它构建的是一个作用于 num_qubits 个比特的局部算子。
        """
        # --- 步骤 1: 严格的输入验证 (在此局部上下文中) ---
        if not isinstance(num_qubits, int) or num_qubits < 1:
            self._internal_logger.error(f"_build_local_mcx_operator: 'num_qubits' must be a positive integer, got {num_qubits}.")
            raise ValueError("'num_qubits' must be a positive integer.")
            
        all_qubits = controls + [target]
        if not all(isinstance(q, int) and 0 <= q < num_qubits for q in all_qubits):
            self._internal_logger.error(f"_build_local_mcx_operator: All control/target qubit indices must be valid for a {num_qubits}-qubit system. Got {all_qubits}.")
            raise ValueError("All control/target qubit indices must be valid within the local qubit space.")

        if len(set(all_qubits)) != len(all_qubits):
            self._internal_logger.error(f"_build_local_mcx_operator: All qubits (controls and target) must be distinct. Got controls={controls}, target={target}.")
            raise ValueError("All qubits (controls and target) for the local MCX operator must be distinct.")
            
        # --- 步骤 2: 初始化和准备 ---
        op_dim = 1 << num_qubits
        mcx_local_u = self._backend.eye(op_dim, dtype=complex)
        
        control_mask = sum(1 << q for q in controls)
        target_mask = 1 << target

        # --- 步骤 3: 根据后端类型构建矩阵 ---
        if isinstance(mcx_local_u, list): # PurePythonBackend
            # 遍历所有可能的局部基态索引
            for i in range(op_dim):
                # 检查是否满足所有控制条件，并且为了避免重复操作，只处理目标比特为 0 的情况
                if (i & control_mask) == control_mask and (i & target_mask) == 0:
                    # i 是 |...1...1_controls...0_target...⟩
                    j = i | target_mask
                    
                    # 在单位矩阵中执行交换操作
                    # |i⟩<i| 和 |j⟩<j| 变为 0, |i⟩<j| 和 |j⟩<i| 变为 1
                    mcx_local_u[i][i] = 0.0 + 0.0j
                    mcx_local_u[j][j] = 0.0 + 0.0j
                    mcx_local_u[i][j] = 1.0 + 0.0j
                    mcx_local_u[j][i] = 1.0 + 0.0j
        else: # CuPy/NumPy Backend
            # 1. 创建一个从 0 到 op_dim-1 的索引数组
            indices = self._backend.arange(op_dim)
            
            # 2. 找到所有满足“控制位为1”且“目标位为0”的基态索引
            base_indices = indices[((indices & control_mask) == control_mask) & ((indices & target_mask) == 0)]
            
            # 3. 只有在找到了这样的基态时才执行操作
            if hasattr(base_indices, 'size') and base_indices.size > 0:
                # 计算与它们配对的、目标位为1的索引
                swapped_indices = base_indices | target_mask
                
                # 4. 使用高级索引批量地执行交换操作
                # 将 |i⟩<i| 和 |j⟩<j| 的对角线元素清零
                mcx_local_u[base_indices, base_indices] = 0
                mcx_local_u[swapped_indices, swapped_indices] = 0
                
                # 将 |i⟩<j| 和 |j⟩<i| 的非对角线元素置为1
                mcx_local_u[base_indices, swapped_indices] = 1
                mcx_local_u[swapped_indices, base_indices] = 1
                
        return mcx_local_u

    def _build_local_mcz_operator(self, num_qubits: int, controls: List[int], target: int) -> Any:
        """
        [内部辅助方法][新增] 构建一个局部的 Multi-Controlled-Z (MCZ) 酉矩阵。
        """
        op_dim = 1 << num_qubits
        mcz_local_u = self._backend.eye(op_dim, dtype=complex)
        full_mask = sum(1 << q for q in controls) | (1 << target)
        
        if isinstance(mcz_local_u, list):
            for i in range(op_dim):
                if (i & full_mask) == full_mask:
                    mcz_local_u[i][i] *= -1.0
        else:
            indices = self._backend.arange(op_dim)
            target_indices_mask = (indices & full_mask) == full_mask
            mcz_local_u[target_indices_mask, target_indices_mask] *= -1.0
            
        return mcz_local_u

    def _build_local_mcp_operator(self, num_qubits: int, controls: List[int], target: int, angle: float) -> Any:
        """
        [内部辅助方法][新增] 构建一个局部的 Multi-Controlled-Phase (MCP) 酉矩阵。
        """
        op_dim = 1 << num_qubits
        mcp_local_u = self._backend.eye(op_dim, dtype=complex)
        full_mask = sum(1 << q for q in controls) | (1 << target)
        phase = self._backend.cmath.exp(1j * angle)

        if isinstance(mcp_local_u, list):
            for i in range(op_dim):
                if (i & full_mask) == full_mask:
                    mcp_local_u[i][i] *= phase
        else:
            indices = self._backend.arange(op_dim)
            target_indices_mask = (indices & full_mask) == full_mask
            mcp_local_u[target_indices_mask, target_indices_mask] *= phase

        return mcp_local_u
    def _build_local_mcu_operator(self, num_qubits: int, controls: List[int], target: int, u_matrix: List[List[complex]]) -> Any:
        """
        [内部辅助方法][新增] 构建一个局部的 Multi-Controlled-U (MCU) 酉矩阵。
        """
        op_dim = 1 << num_qubits
        mcu_local_u = self._backend.eye(op_dim, dtype=complex)
        u00, u01 = complex(u_matrix[0][0]), complex(u_matrix[0][1])
        u10, u11 = complex(u_matrix[1][0]), complex(u_matrix[1][1])
        control_mask = sum(1 << q for q in controls)
        target_mask = 1 << target

        if isinstance(mcu_local_u, list):
            for i in range(op_dim):
                if (i & control_mask) == control_mask and (i & target_mask) == 0:
                    idx0, idx1 = i, i | target_mask
                    mcu_local_u[idx0][idx0], mcu_local_u[idx0][idx1] = u00, u01
                    mcu_local_u[idx1][idx0], mcu_local_u[idx1][idx1] = u10, u11
        else:
            indices = self._backend.arange(op_dim)
            base_indices = indices[((indices & control_mask) == control_mask) & ((indices & target_mask) == 0)]
            if hasattr(base_indices, 'size') and base_indices.size > 0:
                swapped_indices = base_indices | target_mask
                mcu_local_u[base_indices, base_indices] = u00
                mcu_local_u[base_indices, swapped_indices] = u01
                mcu_local_u[swapped_indices, base_indices] = u10
                mcu_local_u[swapped_indices, swapped_indices] = u11
        
        return mcu_local_u  
    def _build_mcp_operator(self, controls: List[int], target: int, angle: float) -> Any:
        """
        [健壮性与后端优化增强版] 构建 Multi-Controlled-Phase (MCP) 门的全局酉矩阵。

        此方法用于 `run_circuit_on_entity` 的后备路径，当无法使用优化内核时
        （例如，在矩阵累积模式下），需要构建完整的算子矩阵。

        MCP 门（也称 n-CP Gate）是一个对角矩阵，其作用是：当且仅当所有控制比特 `controls`
        和目标比特 `target` 都处于 `|1⟩` 态时，给该计算基矢的振幅施加一个 `e^(i*angle)` 的相位。
        在矩阵表示中，这意味着在对应的对角线位置上，元素值从 1 变为 `e^(i*angle)`。

        核心增强功能:
        - 增加了对 `controls` 列表、`target` 和 `angle` 的全面验证。
        - 实现了后端感知的优化：对于 CuPy/NumPy 后端，使用高效的向量化和布尔索引
          来批量修改对角线元素；对于 PurePython 后端，使用清晰的 for 循环。
        - 增加了详尽的文档和内部注释。

        Args:
            controls (List[int]): 控制量子比特的索引列表。
            target (int): 目标量子比特的索引。
            angle (float): 旋转角度 (单位：弧度)。

        Returns:
            Any: MCP 门的全局酉矩阵，其类型与当前后端匹配。

        Raises:
            ValueError: 如果 `controls` 或 `target` 包含无效或重叠的量子比特索引。
            TypeError: 如果 `controls`、`target` 或 `angle` 的类型不正确。
        """
        log_prefix = f"[_StateVectorEntity._build_mcp_operator(N={self.num_qubits})]"

        # --- 步骤 1: 极度严格的输入验证 ---
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"{log_prefix} Invalid 'controls' list {controls}. Must be a list of valid qubit indices.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"{log_prefix} Invalid 'target' qubit {target}. Must be a valid qubit index.")
            raise ValueError(f"Invalid target qubit index {target}.")
        
        if target in controls:
            self._internal_logger.error(f"{log_prefix} Target qubit ({target}) cannot be in the controls list ({controls}).")
            raise ValueError("Target qubit for MCP gate cannot be in the controls list.")
        
        if not isinstance(angle, (float, int)):
            self._internal_logger.error(f"{log_prefix} 'angle' must be a numeric type, but got {type(angle).__name__}.")
            raise TypeError("Angle must be a numeric type for MCP gate.")
            
        if len(set(controls)) != len(controls):
             self._internal_logger.error(f"{log_prefix} 'controls' list contains duplicate qubit indices: {controls}.")
             raise ValueError("Controls list for MCP gate must not contain duplicate qubit indices.")

        self._internal_logger.debug(f"{log_prefix} Building MCP({angle:.4f}) operator matrix for controls={controls}, target={target}.")

        # --- 步骤 2: 初始化和准备 ---
        op_dim = 1 << self.num_qubits
        # 创建一个单位矩阵作为基础
        mcp_global_u = self._backend.eye(op_dim, dtype=complex)
        
        # 使用位掩码技术高效检查条件。
        # 创建一个掩码，其中所有控制比特和目标比特对应的位都为 1。
        full_mask = sum(1 << q for q in controls) | (1 << target)
        
        # 使用后端抽象的数学函数获取相位因子 e^(i*angle)
        phase = self._backend.cmath.exp(1j * angle)
        
        # --- 步骤 3: 根据后端类型构建矩阵 ---
        if isinstance(mcp_global_u, list):
            # --- 路径 A: PurePythonBackend ---
            # 遍历所有对角线元素
            for i in range(op_dim):
                # 检查当前索引 `i` 是否与 `full_mask` 完全匹配。
                # 这等效于检查 `i` 中所有控制位和目标位是否都为 1。
                if (i & full_mask) == full_mask:
                    # 如果匹配，将该对角线元素乘以相位因子
                    mcp_global_u[i][i] *= phase
        else:
            # --- 路径 B: CuPy/NumPy Backend (使用向量化操作) ---
            # 1. 创建一个从 0 到 op_dim-1 的索引数组
            indices = self._backend.arange(op_dim)
            
            # 2. 创建一个布尔掩码，标记所有满足条件的对角线位置
            target_indices_mask = (indices & full_mask) == full_mask
            
            # 3. 使用高级布尔索引来一次性地修改所有目标对角线元素
            mcp_global_u[target_indices_mask, target_indices_mask] *= phase
        
        return mcp_global_u

    def _build_mcu_operator(self, controls: List[int], target: int, u_matrix: List[List[complex]]) -> Any:
        """
        [健壮性与后端优化增强版] 构建 Multi-Controlled-U (MCU) 门的全局酉矩阵。

        此方法用于 `run_circuit_on_entity` 的后备路径，当无法使用优化内核时
        （例如，在矩阵累积模式下），需要构建完整的算子矩阵。

        MCU 门的作用是：当且仅当所有控制比特 `controls` 都处于 `|1⟩` 态时，
        在目标比特 `target` 上应用一个单比特酉矩阵 `u_matrix`。在矩阵表示中，
        这意味着对于所有控制位为1的基态，对应的 2x2 子矩阵被 `u_matrix` 替换，
        而其他地方则保持单位矩阵的形式。

        核心增强功能:
        - 增加了对所有输入参数的全面验证。
        - 实现了后端感知的优化：对于 CuPy/NumPy 后端，使用高效的向量化和高级索引
          来批量地嵌入 2x2 矩阵；对于 PurePython 后端，使用清晰的 for 循环。
        - 增加了详尽的文档和内部注释。

        Args:
            controls (List[int]): 控制量子比特的索引列表。
            target (int): 目标量子比特的索引。
            u_matrix (List[List[complex]]): 要应用的 2x2 单比特酉矩阵。

        Returns:
            Any: MCU 门的全局酉矩阵，其类型与当前后端匹配。

        Raises:
            ValueError: 如果 `controls` 或 `target` 包含无效或重叠的量子比特索引，
                        或 `u_matrix` 格式不正确。
            TypeError: 如果输入参数类型不正确。
        """
        log_prefix = f"[_StateVectorEntity._build_mcu_operator(N={self.num_qubits})]"

        # --- 步骤 1: 极度严格的输入验证 ---
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"{log_prefix} Invalid 'controls' list {controls}. Must be a list of valid qubit indices.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"{log_prefix} Invalid 'target' qubit {target}. Must be a valid qubit index.")
            raise ValueError(f"Invalid target qubit index {target}.")
        
        if target in controls:
            self._internal_logger.error(f"{log_prefix} Target qubit ({target}) cannot be in the controls list ({controls}).")
            raise ValueError("Target qubit for MCU gate cannot be in the controls list.")
        
        if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
                isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
                isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2 and
                all(isinstance(el, (complex, float, int)) for row in u_matrix for el in row)):
            self._internal_logger.error(f"{log_prefix} 'u_matrix' must be a 2x2 nested list of complex numbers. Got {u_matrix}.")
            raise ValueError("`u_matrix` for MCU must be a 2x2 nested list of complex numbers.")
            
        if len(set(controls)) != len(controls):
             self._internal_logger.error(f"{log_prefix} 'controls' list contains duplicate qubit indices: {controls}.")
             raise ValueError("Controls list for MCU gate must not contain duplicate qubit indices.")

        self._internal_logger.debug(f"{log_prefix} Building MCU operator matrix for controls={controls}, target={target}.")

        # --- 步骤 2: 初始化和准备 ---
        op_dim = 1 << self.num_qubits
        # 创建一个单位矩阵作为基础
        mcu_global_u = self._backend.eye(op_dim, dtype=complex)
        
        # 从 u_matrix 中提取元素
        u00, u01 = complex(u_matrix[0][0]), complex(u_matrix[0][1])
        u10, u11 = complex(u_matrix[1][0]), complex(u_matrix[1][1])
        
        # 创建控制位掩码和目标位掩码
        full_control_mask = sum(1 << q for q in controls)
        target_mask = 1 << target
        
        # --- 步骤 3: 根据后端类型构建矩阵 ---
        if isinstance(mcu_global_u, list):
            # --- 路径 A: PurePythonBackend ---
            # 遍历所有可能的基态索引
            for i in range(op_dim):
                # 检查是否满足所有控制条件，并且为了避免重复操作，只处理目标比特为 0 的情况
                if (i & full_control_mask) == full_control_mask and (i & target_mask) == 0:
                    # i 是 |...1...1_controls...0_target...⟩
                    idx0 = i
                    # j 是 |...1...1_controls...1_target...⟩
                    idx1 = i | target_mask
                    
                    # 在全局单位矩阵中，将对应的 2x2 对角块替换为 u_matrix
                    mcu_global_u[idx0][idx0] = u00
                    mcu_global_u[idx0][idx1] = u01
                    mcu_global_u[idx1][idx0] = u10
                    mcu_global_u[idx1][idx1] = u11
        else:
            # --- 路径 B: CuPy/NumPy Backend (使用向量化操作) ---
            # 1. 创建一个从 0 到 op_dim-1 的索引数组
            indices = self._backend.arange(op_dim)
            
            # 2. 找到所有满足“控制位为1”且“目标位为0”的基态索引
            base_indices = indices[((indices & full_control_mask) == full_control_mask) & ((indices & target_mask) == 0)]
            
            # 3. 只有在找到了这样的基态时才执行操作
            if hasattr(base_indices, 'size') and base_indices.size > 0:
                # 计算与它们配对的、目标位为1的索引
                swapped_indices = base_indices | target_mask
                
                # 4. 使用高级索引批量地嵌入 u_matrix 的四个元素
                mcu_global_u[base_indices, base_indices] = u00
                mcu_global_u[base_indices, swapped_indices] = u01
                mcu_global_u[swapped_indices, base_indices] = u10
                mcu_global_u[swapped_indices, swapped_indices] = u11
                
        return mcu_global_u
    
    
    
    # --- 核心方法: 在实体上运行电路 ---
   
    def run_circuit_on_entity(self, circuit: 'QuantumCircuit'):
        """
        [最终修复版 - 无递归] 在态矢量实体上执行量子线路。
        """
        if not hasattr(circuit, 'num_qubits') or not hasattr(circuit, 'instructions'):
            raise TypeError("Input 'circuit' must be a QuantumCircuit-like object.")

        if self.num_qubits < circuit.num_qubits:
            raise ValueError("Entity's qubit count is less than circuit's qubit count.")

        log_prefix = f"[_StateVectorEntity.run_circuit_on_entity(N={self.num_qubits})]"
        
        is_matrix_mode = (hasattr(self._state_vector, 'ndim') and self._state_vector.ndim == 2) or \
                        (isinstance(self._state_vector, list) and self._state_vector and isinstance(self._state_vector[0], list))
        
        mode_str = '矩阵累积' if is_matrix_mode else '态矢量'
        self._internal_logger.debug(f"[{log_prefix}] Starting circuit execution in '{mode_str}' mode.")

        for instr_index, instruction in enumerate(circuit.instructions):
            try:
                gate_name = instruction[0]
                op_kwargs = instruction[-1] if isinstance(instruction[-1], dict) else {}
                op_args = list(instruction[1:-1]) if isinstance(instruction[-1], dict) else list(instruction[1:])

                if gate_name in ('barrier', 'simulate_measurement', 'apply_quantum_channel'):
                    continue
                if 'condition' in op_kwargs:
                    continue
                    
                op_def = QuantumOperatorLibrary.get(gate_name)
                if op_def is None:
                    raise ValueError(f"Unknown gate '{gate_name}'.")

                # --- [最终逻辑分派] ---
                # 路径 1: 优化内核
                if not is_matrix_mode and op_def.sv_kernel_name:
                    kernel_func = getattr(self, op_def.sv_kernel_name, None)
                    if kernel_func and callable(kernel_func):
                        self._internal_logger.debug(f"[{log_prefix}] Applying kernel for '{gate_name}'.")
                        kernel_func(*op_args, **op_kwargs)
                        continue

                # 路径 2: 宏展开 (仅限真正的复合宏)
                # [CORE FIX] 只有当门没有优化内核时，才考虑分解
                if not op_def.sv_kernel_name and op_def.decomposition_rule:
                    self._internal_logger.debug(f"[{log_prefix}] Expanding composite gate '{gate_name}'.")
                    temp_circuit_for_decomp = QuantumCircuit(self.num_qubits)
                    op_def.decomposition_rule(temp_circuit_for_decomp, *op_args, **op_kwargs)
                    self.run_circuit_on_entity(temp_circuit_for_decomp)
                    continue

                # 路径 3: 通用矩阵构建 (最终后备)
                self._internal_logger.debug(f"[{log_prefix}] Using matrix fallback for '{gate_name}'.")
                
                # --- [CORE FIX] 直接在这里构建局部酉矩阵，不再依赖 get_unitary 的复杂逻辑 ---
                local_unitary = self._get_local_op_for_gate(gate_name, op_args)
                if local_unitary is None:
                    # 如果 _get_local_op_for_gate 无法构建，才尝试 get_unitary
                    # 这为参数化门（如rx,rz）提供了路径
                    params_for_generator = [arg for arg in op_args if isinstance(arg, (float, int, list)) and not (isinstance(arg, int) and 0 <= arg < self.num_qubits)]
                    if op_def.unitary_generator:
                        try:
                           # 提取非比特的参数
                           num_params_expected = len(op_def.parameter_info) if op_def.parameter_info else 1
                           params_only = op_args[-num_params_expected:]
                           local_unitary = op_def.get_unitary(*params_only, **op_kwargs)
                        except (ValueError, TypeError) as e:
                             raise RuntimeError(f"Failed to generate unitary for '{gate_name}' via generator: {e}") from e

                if local_unitary is None:
                     raise RuntimeError(f"Could not obtain a unitary matrix for '{gate_name}'.")

                # 确定目标比特
                all_qubits = [q for arg in op_args for q in (arg if isinstance(arg, list) else [arg]) if isinstance(q, int)]
                # 使用 op_def.num_qubits 来确定正确的比特列表和顺序
                if op_def.num_qubits > 0: # 固定比特数门
                    target_qubits = [q for q in op_args if isinstance(q, int)][:op_def.num_qubits]
                else: # 可变比特门 (约定： controls, target)
                    target_qubits = op_args[0] + [op_args[1]]

                global_unitary = self._build_global_operator_multi_qubit(target_qubits, local_unitary)
                self._apply_global_unitary(global_unitary)
            
            except Exception as e:
                self._internal_logger.critical(f"[{log_prefix}] Failed to execute instruction #{instr_index} ('{instruction}'): {e}", exc_info=True)
                raise RuntimeError(f"Execution of instruction '{instruction}' (index {instr_index}) failed.") from e

        self._internal_logger.debug(f"[{log_prefix}] Circuit execution completed successfully.")
    
    
    def _get_local_op_from_library(self, gate_name: str, op_args: List[Any], op_kwargs: Dict[str, Any]) -> Optional[List[List[complex]]]:
        """
        [最终完整版 - 基于 OperatorLibrary] 从中央库中获取指定门操作的局部酉矩阵。

        此方法作为 `run_circuit_on_entity` 中通用矩阵构建路径的核心辅助函数。
        它完全委托给 QuantumOperatorLibrary 和 OperatorDefinition.get_unitary 方法
        来获取与后端无关的、纯Python列表形式的酉矩阵。

        工作流程:
        1.  从 QuantumOperatorLibrary 中查询 `gate_name` 对应的 OperatorDefinition。
        2.  如果找到了定义，则调用该定义的 `get_unitary` 方法，并将门操作的
            参数 `op_args` 和 `op_kwargs` 传递给它。
        3.  `get_unitary` 方法会智能地处理三种情况：
            a.  返回一个静态定义的 `unitary_matrix`。
            b.  调用 `unitary_generator` 函数，用传入的参数动态生成矩阵。
            c.  (作为昂贵的后备) 调用 `decomposition_rule`，在临时电路上展开宏，
                然后计算其等效的酉矩阵。
        4.  返回最终获取的、后端无关的酉矩阵。

        Args:
            gate_name (str): 门的名称。
            op_args (List[Any]): 门的位置参数。
            op_kwargs (Dict[str, Any]): 门的关键字参数。

        Returns:
            Optional[List[List[complex]]]: 
                一个代表门局部酉矩阵的 Python 嵌套列表。如果门无法被识别或
                构建失败，则返回 `None`。
        
        Raises:
            ValueError: 如果从库中找不到门定义，或者参数与定义不匹配。
            RuntimeError: 如果在通过分解规则计算酉矩阵时发生内部错误。
        """
        log_prefix = f"[_StateVectorEntity._get_local_op_from_library(Gate='{gate_name}')]"
        
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(gate_name, str) or not gate_name.strip():
            self._internal_logger.error(f"[{log_prefix}] Invalid 'gate_name'. Must be a non-empty string.")
            # 对于内部方法，可以选择返回 None 或抛出异常。抛出异常更明确。
            raise TypeError("gate_name must be a non-empty string.")
        
        # --- 步骤 2: 从库中查询操作定义 ---
        op_def = QuantumOperatorLibrary.get(gate_name)
        
        if op_def is None:
            self._internal_logger.error(f"[{log_prefix}] No definition found in QuantumOperatorLibrary for gate '{gate_name}'. Cannot build operator.")
            # 明确地表示无法构建
            raise ValueError(f"No definition found in QuantumOperatorLibrary for gate '{gate_name}'.")

        self._internal_logger.debug(f"[{log_prefix}] Found definition for '{gate_name}'. Proceeding to get unitary representation.")

        # --- 步骤 3: 委托给 OperatorDefinition.get_unitary 方法 ---
        try:
            # 将所有复杂的逻辑（静态、生成器、分解）都委托给 op_def 对象自身
            local_unitary = op_def.get_unitary(*op_args, **op_kwargs)
            
            # --- 步骤 4: 验证返回结果 ---
            if local_unitary is None:
                # 这种情况理论上不应该发生，因为 get_unitary 失败时会抛出异常
                self._internal_logger.error(f"[{log_prefix}] op_def.get_unitary for '{gate_name}' unexpectedly returned None.")
                return None
            
            # 做一个基本的格式检查
            if not isinstance(local_unitary, list) or (local_unitary and not isinstance(local_unitary[0], list)):
                self._internal_logger.error(f"[{log_prefix}] op_def.get_unitary for '{gate_name}' returned a non-matrix-like object of type {type(local_unitary).__name__}.")
                raise TypeError(f"Unitary representation for '{gate_name}' must be a nested list (matrix).")
            
            self._internal_logger.debug(f"[{log_prefix}] Successfully obtained a {len(local_unitary)}x{len(local_unitary[0]) if local_unitary else 0} local unitary matrix for '{gate_name}'.")
            
            return local_unitary

        except (ValueError, TypeError, RuntimeError, NotImplementedError) as e:
            # 捕获来自 get_unitary 的所有已知异常
            self._internal_logger.critical(f"[{log_prefix}] Failed to get local operator for '{gate_name}' with args {op_args}: {e}", exc_info=True)
            # 将异常重新包装，以提供更清晰的上下文
            raise RuntimeError(f"Failed to build local operator for gate '{gate_name}'.") from e
        except Exception as e_unhandled:
            # 捕获任何其他意外的异常
            self._internal_logger.critical(f"[{log_prefix}] An unexpected error occurred while getting local operator for '{gate_name}': {e_unhandled}", exc_info=True)
            raise RuntimeError(f"Unexpected error while building local operator for '{gate_name}'.") from e_unhandled
    
    # --- [NEW] 添加一个用于构建本地酉算子的辅助方法 (用于通用矩阵回退路径) ---
    def _get_local_op_for_gate(self, gate_name: str, op_args: List[Any]) -> Optional[Any]:
        """
        [内部辅助方法][最终正确性修复版 v4 - 多控门参数解析修复] 根据门名称和参数，
        生成其在局部量子比特上作用的酉矩阵。
        """
        log_prefix = f"[_StateVectorEntity._get_local_op_for_gate(Gate='{gate_name}')]"
        
        try:
            global _get_effective_unitary_placeholder
            
            import math
            import cmath

            # --- 基础单比特门 ---
            if gate_name == 'x': return self._SIGMA_X_LIST
            if gate_name == 'y': return self._SIGMA_Y_LIST
            if gate_name == 'z': return self._SIGMA_Z_LIST
            if gate_name == 'h':
                sqrt2_inv = 1.0 / math.sqrt(2.0)
                return [[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]]
            if gate_name == 's': return [[1+0j, 0+0j], [0+0j, 1j]]
            if gate_name == 'sdg': return [[1+0j, 0+0j], [0+0j, -1j]]
            if gate_name == 't_gate': return [[1+0j, 0+0j], [0+0j, cmath.exp(1j * math.pi / 4.0)]]
            if gate_name == 'tdg': return [[1+0j, 0+0j], [0+0j, cmath.exp(-1j * math.pi / 4.0)]]
            if gate_name == 'sx':
                return [[(1+1j)/2, (1-1j)/2], [(1-1j)/2, (1+1j)/2]]

            # --- 参数化单比特门 ---
            if gate_name == 'rx' and len(op_args) >= 2:
                theta = op_args[1]
                half_theta = theta / 2.0
                c = math.cos(half_theta)
                s = math.sin(half_theta)
                return [[c, -1j*s], [-1j*s, c]]
            if gate_name == 'ry' and len(op_args) >= 2:
                theta = op_args[1]
                half_theta = theta / 2.0
                c = math.cos(half_theta)
                s = math.sin(half_theta)
                return [[c, -s], [s, c]]
            if gate_name == 'rz' and len(op_args) >= 2:
                phi = op_args[1]
                half_phi = phi / 2.0
                return [[cmath.exp(-1j * half_phi), 0+0j], [0+0j, cmath.exp(1j * half_phi)]]
            if gate_name == 'p_gate' and len(op_args) >= 2:
                lambda_angle = op_args[1]
                return [[1+0j, 0+0j], [0+0j, cmath.exp(1j * lambda_angle)]]
            if gate_name == 'u3_gate' and len(op_args) >= 4:
                theta, phi, lambda_angle = op_args[1], op_args[2], op_args[3]
                c_half_theta = math.cos(theta/2)
                s_half_theta = math.sin(theta/2)
                return [[c_half_theta, -cmath.exp(1j*lambda_angle)*s_half_theta],
                        [cmath.exp(1j*phi)*s_half_theta, cmath.exp(1j*(phi+lambda_angle))*c_half_theta]]

            # --- 双比特门 ---
            if gate_name == 'cnot' and len(op_args) >= 2:
                return [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
            if gate_name == 'cz' and len(op_args) >= 2:
                return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]]
            if gate_name == 'cp' and len(op_args) >= 3:
                angle = op_args[2]
                phase = cmath.exp(1j * angle)
                return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,phase]]
            if gate_name == 'crx' and len(op_args) >= 3:
                theta = op_args[2]
                half_theta = theta / 2.0
                c = math.cos(half_theta)
                s = math.sin(half_theta)
                return [[1,0,0,0],[0,1,0,0],[0,0,c,-1j*s],[0,0,-1j*s,c]]
            if gate_name == 'cry' and len(op_args) >= 3:
                theta = op_args[2]
                half_theta = theta / 2.0
                c = math.cos(half_theta)
                s = math.sin(half_theta)
                return [[1,0,0,0],[0,1,0,0],[0,0,c,-s],[0,0,s,c]]
            if gate_name == 'crz' and len(op_args) >= 3:
                phi = op_args[2]
                half_phi = phi / 2.0
                phase_0 = cmath.exp(-1j * half_phi)
                phase_1 = cmath.exp(1j * half_phi)
                return [[1,0,0,0],[0,1,0,0],[0,0,phase_0,0],[0,0,0,phase_1]]
            if gate_name == 'controlled_u' and len(op_args) >= 3:
                u_matrix_2x2 = op_args[2]
                u00, u01 = u_matrix_2x2[0]
                u10, u11 = u_matrix_2x2[1]
                return [[1,0,0,0],[0,1,0,0],[0,0,u00,u01],[0,0,u10,u11]]
            if gate_name == 'rxx' and len(op_args) >= 3:
                theta = op_args[2]
                half_theta = theta / 2.0
                c = math.cos(half_theta)
                s_factor = -1j * math.sin(half_theta)
                return [[c,0,0,s_factor],[0,c,s_factor,0],[0,s_factor,c,0],[s_factor,0,0,c]]
            if gate_name == 'ryy' and len(op_args) >= 3:
                theta = op_args[2]
                half_theta = theta / 2.0
                c = math.cos(half_theta)
                s_factor_pos = 1j * math.sin(half_theta)
                s_factor_neg = -1j * math.sin(half_theta)
                return [[c,0,0,s_factor_pos],[0,c,s_factor_neg,0],[0,s_factor_neg,c,0],[s_factor_pos,0,0,c]]
            if gate_name == 'rzz' and len(op_args) >= 3:
                theta = op_args[2]
                half_theta = theta / 2.0
                phase_even = cmath.exp(-1j * half_theta)
                phase_odd = cmath.exp(1j * half_theta)
                return [[phase_even,0,0,0],[0,phase_odd,0,0],[0,0,phase_odd,0],[0,0,0,phase_even]]
            if gate_name == 'iswap' and len(op_args) >= 2:
                return [[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]

            # --- 复合门 ---
            if gate_name in ['ecr', 'ecrdg', 'fredkin', 'swap']:
                 if _get_effective_unitary_placeholder is None:
                     raise RuntimeError("_get_effective_unitary_placeholder not set, cannot compute unitary for complex gates.")
                 
                 macro_info = AlgorithmBuilders.get_macro_definition(gate_name)
                 if macro_info:
                     _, num_local_qubits, _ = macro_info
                     temp_qc_for_unitary = QuantumCircuit(num_local_qubits)
                     
                     macro_func = macro_info[0]
                     local_args = list(range(num_local_qubits))
                     macro_func(temp_qc_for_unitary, *local_args, **{})
                     
                     return _get_effective_unitary_placeholder(temp_qc_for_unitary)
            
            # --- 多控制门 ---
            if gate_name == 'toffoli':
                return self._build_local_mcx_operator(num_qubits=3, controls=[2, 1], target=0)
            
            # [FINAL FIX] 辅助函数保持不变，它本身是正确的
            def get_local_indices(qubits: List[int]):
                """将全局比特索引列表映射到局部索引。"""
                all_qubits_indices = sorted(list(set(qubits)))
                num_local = len(all_qubits_indices)
                global_to_local_map = {q_global: q_local for q_local, q_global in enumerate(all_qubits_indices)}
                return num_local, global_to_local_map

            # [FINAL FIX] 正确地从 op_args 中解析多控制门的参数
            if gate_name == 'mcx' and len(op_args) >= 2:
                controls, target = op_args[0], op_args[1]
                # [CORE FIX] 将 target 整数包装在列表中再进行连接
                all_global_qubits = controls + [target]
                num_local, g2l_map = get_local_indices(all_global_qubits)
                local_controls = [g2l_map[q] for q in controls]
                local_target = g2l_map[target]
                return self._build_local_mcx_operator(num_qubits=num_local, controls=local_controls, target=local_target)
            
            if gate_name == 'mcz' and len(op_args) >= 2:
                controls, target = op_args[0], op_args[1]
                all_global_qubits = controls + [target]
                num_local, g2l_map = get_local_indices(all_global_qubits)
                local_controls = [g2l_map[q] for q in controls]
                local_target = g2l_map[target]
                return self._build_local_mcz_operator(num_qubits=num_local, controls=local_controls, target=local_target)

            if gate_name == 'mcp' and len(op_args) >= 3:
                controls, target, angle = op_args[0], op_args[1], op_args[2]
                all_global_qubits = controls + [target]
                num_local, g2l_map = get_local_indices(all_global_qubits)
                local_controls = [g2l_map[q] for q in controls]
                local_target = g2l_map[target]
                return self._build_local_mcp_operator(num_qubits=num_local, controls=local_controls, target=local_target, angle=angle)

            if gate_name == 'mcu' and len(op_args) >= 3:
                controls, target, u_matrix = op_args[0], op_args[1], op_args[2]
                all_global_qubits = controls + [target]
                num_local, g2l_map = get_local_indices(all_global_qubits)
                local_controls = [g2l_map[q] for q in controls]
                local_target = g2l_map[target]
                return self._build_local_mcu_operator(num_qubits=num_local, controls=local_controls, target=local_target, u_matrix=u_matrix)

            self._internal_logger.warning(f"{log_prefix} No local operator definition found for gate '{gate_name}' with args {op_args}. This will return None.")
            return None

        except Exception as e:
            self._internal_logger.error(f"{log_prefix} Error while getting local operator for gate '{gate_name}' with args {op_args}: {e}", exc_info=True)
            return None
    
    
    
    def _build_fredkin_operator(self, control: int, target_1: int, target_2: int) -> Any:
        """
        [内部辅助方法][新增] 构建 Fredkin (CSWAP) 门的全局酉矩阵。

        此方法用于 `run_circuit_on_entity` 的后备路径，当无法使用优化内核时
        （例如，在矩阵累积模式下），需要构建完整的算子矩阵。

        Args:
            control (int): 控制量子比特的索引。
            target_1 (int): 第一个目标量子比特的索引。
            target_2 (int): 第二个目标量子比特的索引。

        Returns:
            Any: Fredkin 门的全局酉矩阵。

        Raises:
            ValueError: 如果 `control`, `target_1`, `target_2` 无效。
        """
        # [健壮性改进] 输入验证
        qubits = [control, target_1, target_2]
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits):
            self._internal_logger.error(f"_build_fredkin_operator: Invalid qubit indices {qubits} for {self.num_qubits} qubits.")
            raise ValueError(f"Invalid qubit index for Fredkin gate.")
        if len(set(qubits)) != 3:
            self._internal_logger.error(f"_build_fredkin_operator: All qubits for Fredkin must be distinct. Got {qubits}.")
            raise ValueError("All qubits for Fredkin gate must be distinct.")

        self._internal_logger.warning(
            "_build_fredkin_operator: Fredkin gate is being built via matrix fallback. This indicates "
            "that the macro expansion path failed, which is unexpected or a performance bottleneck."
        )
        op_dim = 1 << self.num_qubits
        fredkin_global_u = self._backend.eye(op_dim, dtype=complex)
        control_mask = 1 << control
        target1_mask = 1 << target_1
        target2_mask = 1 << target_2

        if isinstance(fredkin_global_u, list): # PurePythonBackend
            for i in range(op_dim):
                if (i & control_mask) != 0: # 检查控制位为1
                    bit1 = (i >> target_1) & 1
                    bit2 = (i >> target_2) & 1
                    if bit1 != bit2: # 且两个目标比特状态不同
                        j = i ^ target1_mask ^ target2_mask # 计算交换后的索引
                        if i < j: # 为了避免对同一对索引操作两次，只在 i < j 时执行
                            fredkin_global_u[i][i] = 0.0 + 0.0j
                            fredkin_global_u[j][j] = 0.0 + 0.0j
                            fredkin_global_u[i][j] = 1.0 + 0.0j
                            fredkin_global_u[j][i] = 1.0 + 0.0j
        else: # CuPy/NumPy Backend
            indices = self._backend.arange(op_dim)
            
            # 找到所有满足“控制位为1”且“目标位不同”条件的基态索引
            base_indices = indices[
                ((indices & control_mask) != 0) &
                (((indices >> target_1) & 1) != ((indices >> target_2) & 1))
            ]
            
            swapped_indices = base_indices ^ target1_mask ^ target2_mask
            # 为了避免重复交换，只取 base_indices < swapped_indices 的部分
            unique_base_indices = base_indices[base_indices < swapped_indices]
            
            if hasattr(unique_base_indices, 'size') and unique_base_indices.size > 0:
                unique_swapped_indices = unique_base_indices ^ target1_mask ^ target2_mask
                
                fredkin_global_u[unique_base_indices, unique_base_indices] = 0
                fredkin_global_u[unique_swapped_indices, unique_swapped_indices] = 0
                fredkin_global_u[unique_base_indices, unique_swapped_indices] = 1
                fredkin_global_u[unique_swapped_indices, unique_base_indices] = 1
                
        return fredkin_global_u

class _DensityMatrixEntity:
    """
    [最终完整版] 一个内部辅助类，代表一个“实体化”的量子态，基于密度矩阵。
    它直接存储和操作密度矩阵，并为常用门和噪声通道提供了优化内核，
    以显著提升带噪模拟的性能。
    """
    def __init__(self, num_qubits: int, backend: Any, initial_dm: Optional[Any] = None):
        """
        初始化密度矩阵实体。
        [v1.5.11 修正版] 在内核映射中添加了对 'apply_quantum_channel' 的支持。
        """
        self._internal_logger = logging.getLogger(f"{_DensityMatrixEntity.__module__}.{_DensityMatrixEntity.__name__}")
        
        if not isinstance(num_qubits, int) or num_qubits < 0:
            raise ValueError("num_qubits must be a non-negative integer.")
        if backend is None:
            raise TypeError("A valid backend instance must be provided.")
        
        self.num_qubits = num_qubits
        self._backend = backend
        
        if initial_dm is not None:
            self._density_matrix = initial_dm
        else:
            dim = 1 << num_qubits
            self._density_matrix = self._backend.zeros((dim, dim), dtype=complex)
            if dim > 0:
                if isinstance(self._density_matrix, list):
                    self._density_matrix[0][0] = 1.0 + 0.0j
                else:
                    self._density_matrix[0, 0] = 1.0 + 0.0j

        # --- [核心修复] ---
        # 将 'apply_quantum_channel' 指令映射到其对应的内核实现
        self._kernel_map: Dict[str, Callable[..., None]] = {
            'x': self._apply_x_kernel_dm,
            'y': self._apply_y_kernel_dm,
            'z': self._apply_z_kernel_dm,
            'h': self._apply_h_kernel_dm,
            's': self._apply_s_kernel_dm,
            'sdg': self._apply_sdg_kernel_dm,
            't_gate': self._apply_t_gate_kernel_dm,
            'tdg': self._apply_tdg_kernel_dm,
            'rx': self._apply_rx_kernel_dm,
            'ry': self._apply_ry_kernel_dm,
            'rz': self._apply_rz_kernel_dm,
            'p_gate': self._apply_p_gate_kernel_dm,
            'cnot': self._apply_cnot_kernel_dm,
            'cz': self._apply_cz_kernel_dm,
            'cp': self._apply_cp_kernel_dm,
            'crx': self._apply_crx_kernel_dm,
            'cry': self._apply_cry_kernel_dm,
            'crz': self._apply_crz_kernel_dm,
            # 新增的映射，将高层指令 'apply_quantum_channel' 链接到其内核实现
            'apply_quantum_channel': self._apply_channel_kernel_dm,
        }
        # --- [修复结束] ---
        
        self._internal_logger.debug(f"[_DensityMatrixEntity.__init__(N={self.num_qubits})] Initialized successfully.")
    def normalize(self):
        """对密度矩阵进行归一化，确保其迹为1。"""
        trace = self._backend.trace(self._density_matrix)
        trace_float = float(trace.real)
        if not self._backend.isclose(trace_float, 1.0, atol=1e-9):
            if trace_float > 1e-12:
                if isinstance(self._density_matrix, list):
                    self._density_matrix = [[elem / trace for elem in row] for row in self._density_matrix]
                else:
                    self._density_matrix /= trace

    def _apply_x_kernel_dm(self, target_qubit: int):
        """[内核] 高效应用 Pauli-X 门: ρ' = XρX"""
        if not (0 <= target_qubit < self.num_qubits): raise ValueError(f"Invalid target qubit: {target_qubit}")
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        mask = 1 << target_qubit
        
        new_rho = self._backend.zeros_like(rho)
        is_list_backend = isinstance(rho, list)

        for i in range(dim):
            for j in range(dim):
                i_flipped, j_flipped = i ^ mask, j ^ mask
                if is_list_backend:
                    new_rho[i][j] = rho[i_flipped][j_flipped]
                else: # CuPy
                    new_rho[i, j] = rho[i_flipped, j_flipped]
        self._density_matrix = new_rho
        self.normalize()

    def _apply_y_kernel_dm(self, target_qubit: int):
        """[内核] 高效应用 Pauli-Y 门: ρ' = YρY"""
        if not (0 <= target_qubit < self.num_qubits): raise ValueError(f"Invalid target qubit: {target_qubit}")
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        mask = 1 << target_qubit
        new_rho = self._backend.zeros_like(rho)
        is_list_backend = isinstance(rho, list)

        for i in range(dim):
            for j in range(dim):
                sign = -1.0 if ((i >> target_qubit) & 1) != ((j >> target_qubit) & 1) else 1.0
                i_flipped, j_flipped = i ^ mask, j ^ mask
                if is_list_backend:
                    new_rho[i][j] = sign * rho[i_flipped][j_flipped]
                else: # CuPy
                    new_rho[i, j] = sign * rho[i_flipped, j_flipped]
        self._density_matrix = new_rho
        self.normalize()

    def _apply_z_kernel_dm(self, target_qubit: int):
        """[内核] 高效应用 Pauli-Z 门: ρ' = ZρZ (原地修改)"""
        if not (0 <= target_qubit < self.num_qubits): raise ValueError(f"Invalid target qubit: {target_qubit}")
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        is_list_backend = isinstance(rho, list)
        
        for i in range(dim):
            for j in range(dim):
                bit_i = (i >> target_qubit) & 1
                bit_j = (j >> target_qubit) & 1
                if bit_i != bit_j:
                    if is_list_backend:
                        rho[i][j] *= -1.0
                    else: # CuPy
                        rho[i, j] *= -1.0
        self.normalize()

    def _apply_h_kernel_dm(self, target_qubit: int):
        """[内核] 高效应用 Hadamard 门: ρ' = HρH"""
        if not (0 <= target_qubit < self.num_qubits): raise ValueError(f"Invalid target qubit: {target_qubit}")
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        mask = 1 << target_qubit
        new_rho = self._backend.zeros_like(rho)
        is_list_backend = isinstance(rho, list)
        sqrt2_inv = 0.5 # H*H = 1/2 * ...

        for i in range(dim):
            for j in range(dim):
                i_f, j_f = i ^ mask, j ^ mask
                term1 = rho[i][j]
                term2 = rho[i][j_f]
                term3 = rho[i_f][j]
                term4 = rho[i_f][j_f]
                if is_list_backend:
                    new_rho[i][j] = sqrt2_inv * (term1 + term2 + term3 + term4)
                else: # CuPy
                    new_rho[i, j] = sqrt2_inv * (term1 + term2 + term3 + term4)
        self._density_matrix = new_rho
        self.normalize()

    def _apply_phase_gate_kernel_dm(self, target_qubit: int, angle: float):
        """[内核] 通用相位门内核: ρ'_{ij} = e^(i*angle*(bit_i-bit_j)) * ρ_{ij} (原地修改)"""
        if not (0 <= target_qubit < self.num_qubits): raise ValueError(f"Invalid target qubit: {target_qubit}")
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        is_list_backend = isinstance(rho, list)

        phase = self._backend.cmath.exp(1j * angle)
        phase_conj = phase.conjugate()

        for i in range(dim):
            for j in range(dim):
                bit_i = (i >> target_qubit) & 1
                bit_j = (j >> target_qubit) & 1
                
                factor = 1.0
                if bit_i == 1 and bit_j == 0: factor = phase
                elif bit_i == 0 and bit_j == 1: factor = phase_conj
                
                if factor != 1.0:
                    if is_list_backend: rho[i][j] *= factor
                    else: rho[i,j] *= factor
        self.normalize()
        
    def _apply_s_kernel_dm(self, target_qubit: int): self._apply_phase_gate_kernel_dm(target_qubit, math.pi / 2)
    def _apply_sdg_kernel_dm(self, target_qubit: int): self._apply_phase_gate_kernel_dm(target_qubit, -math.pi / 2)
    def _apply_t_gate_kernel_dm(self, target_qubit: int): self._apply_phase_gate_kernel_dm(target_qubit, math.pi / 4)
    def _apply_tdg_kernel_dm(self, target_qubit: int): self._apply_phase_gate_kernel_dm(target_qubit, -math.pi / 4)
    def _apply_p_gate_kernel_dm(self, target_qubit: int, lambda_angle: float): self._apply_phase_gate_kernel_dm(target_qubit, lambda_angle)
    def _apply_rz_kernel_dm(self, target_qubit: int, phi: float):
        """[核心修正] RZ 门: ρ' = RZ(φ)ρRZ(-φ) (原地修改)"""
        if not (0 <= target_qubit < self.num_qubits): raise ValueError(f"Invalid target qubit: {target_qubit}")
        
        # [BUGFIX] 移除了之前错误的、多余的 _apply_phase_gate_kernel_dm 调用
        
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        half_phi = phi / 2.0
        phase_pos = self._backend.cmath.exp(1j * half_phi)
        phase_neg = self._backend.cmath.exp(-1j * half_phi)

        for i in range(dim):
            for j in range(dim):
                bit_i = (i >> target_qubit) & 1
                bit_j = (j >> target_qubit) & 1
                
                factor_i = phase_pos if bit_i == 1 else phase_neg
                factor_j_conj = phase_neg if bit_j == 1 else phase_pos 
                
                if isinstance(rho, list):
                    rho[i][j] *= (factor_i * factor_j_conj)
                else: # CuPy
                    rho[i, j] *= (factor_i * factor_j_conj)
        self.normalize()

    def _apply_rotation_kernel_dm(self, target_qubit: int, theta: float, pauli: str, condition_qubit: Optional[int] = None):
        """
        [核心修正] 通用单比特旋转门内核 (RX, RY)。
        """
        if not (0 <= target_qubit < self.num_qubits): raise ValueError(f"Invalid target qubit: {target_qubit}")
        if condition_qubit is not None and (not (0 <= condition_qubit < self.num_qubits) or condition_qubit == target_qubit):
            raise ValueError(f"Invalid condition qubit: {condition_qubit}")

        dim = 1 << self.num_qubits
        rho = self._density_matrix
        target_mask = 1 << target_qubit
        control_mask = (1 << condition_qubit) if condition_qubit is not None else 0
        
        new_rho = self._backend.zeros_like(rho)
        is_list_backend = isinstance(rho, list)

        c = self._backend.math.cos(theta / 2.0)
        s = self._backend.math.sin(theta / 2.0)
        c2, s2, cs = c*c, s*s, c*s

        for i in range(dim):
            should_apply_row = (condition_qubit is None) or ((i & control_mask) != 0)
            i_f = i ^ target_mask

            for j in range(dim):
                should_apply_col = (condition_qubit is None) or ((j & control_mask) != 0)
                j_f = j ^ target_mask
                
                # [BUGFIX] 条件检查逻辑简化和修正
                if not (should_apply_row and should_apply_col):
                    # 如果任一索引不满足控制条件，则该元素不变
                    if is_list_backend: new_rho[i][j] = rho[i][j]
                    else: new_rho[i, j] = rho[i, j]
                    continue
                
                # [核心修正: 应用正确的更新公式 ρ' = UρU†]
                rho_ij, rho_if_j, rho_i_jf, rho_if_jf = rho[i][j], rho[i_f][j], rho[i][j_f], rho[i_f][j_f]

                val = 0.0 + 0.0j
                if pauli == 'X':
                    val = c2*rho_ij + s2*rho_if_jf - 1j*cs*(rho_if_j - rho_i_jf)
                elif pauli == 'Y':
                    val = c2*rho_ij - s2*rho_if_jf - cs*(rho_if_j + rho_i_jf)
                else: # 默认为单位演化（不应该发生）
                    val = rho_ij

                if is_list_backend: new_rho[i][j] = val
                else: new_rho[i,j] = val
        
        self._density_matrix = new_rho
        self.normalize()
    def _apply_rx_kernel_dm(self, target_qubit: int, theta: float): self._apply_rotation_kernel_dm(target_qubit, theta, 'X')
    def _apply_ry_kernel_dm(self, target_qubit: int, theta: float): self._apply_rotation_kernel_dm(target_qubit, theta, 'Y')

    def _apply_cnot_kernel_dm(self, control: int, target: int):
        """[内核] 高效应用 CNOT 门: ρ' = CNOT * ρ * CNOT"""
        if not (0 <= control < self.num_qubits and 0 <= target < self.num_qubits and control != target):
            raise ValueError("Invalid control/target qubits.")
        
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        control_mask = 1 << control
        target_mask = 1 << target
        new_rho = self._backend.zeros_like(rho)

        for i in range(dim):
            for j in range(dim):
                i_new, j_new = i, j
                if (i & control_mask) != 0: i_new ^= target_mask
                if (j & control_mask) != 0: j_new ^= target_mask
                
                if isinstance(rho, list):
                    new_rho[i_new][j_new] = rho[i][j]
                else: # CuPy
                    new_rho[i_new, j_new] = rho[i, j]

        self._density_matrix = new_rho
        self.normalize()
    
    def _apply_cz_kernel_dm(self, control: int, target: int):
        """
        [内核][v1.5.15 最终正确性与健壮性修复版] 高效应用 Controlled-Z (CZ) 门。
        
        此内核直接在密度矩阵上进行原地修改，避免了构建和应用全局矩阵的开销。
        CZ 门是一个对角矩阵，其作用是：当且仅当控制比特和目标比特都处于 |1⟩ 态时，
        给该计算基矢的振幅施加一个 -1 的相位。
        
        对于密度矩阵，演化规则为 ρ' = UρU†。由于 U 是对角的且 U=U†，
        演化简化为 ρ'_{ij} = U_i * ρ_{ij} * U_j，其中 U_k 是对角线上的第k个元素。
        这等效于当 i 和 j 的比特模式分别翻转相位时，将 ρ_{ij} 乘以相应的相位。

        此版本直接实现了这个演化规则，而不是委托给其他内核，使其成为一个
        完全自包含、独立的优化内核。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。

        Raises:
            TypeError: 如果 `control` 或 `target` 不是整数。
            ValueError: 如果 `control` 或 `target` 无效或相同。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(control, int):
            self._internal_logger.error(f"_apply_cz_kernel_dm: 'control' qubit index must be an integer, but got {type(control).__name__}.")
            raise TypeError("'control' qubit index must be an integer.")
        if not isinstance(target, int):
            self._internal_logger.error(f"_apply_cz_kernel_dm: 'target' qubit index must be an integer, but got {type(target).__name__}.")
            raise TypeError("'target' qubit index must be an integer.")

        if not (0 <= control < self.num_qubits):
            self._internal_logger.error(f"_apply_cz_kernel_dm: Control qubit index {control} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Control qubit index {control} is out of range.")
        if not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"_apply_cz_kernel_dm: Target qubit index {target} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Target qubit index {target} is out of range.")
        
        if control == target:
            self._internal_logger.error(f"_apply_cz_kernel_dm: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits for CZ gate cannot be the same.")

        self._internal_logger.debug(f"Applying CZ kernel on control={control}, target={target}.")

        # --- 步骤 2: 初始化和准备 ---
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        
        # 创建一个位掩码，用于高效检查两个比特是否都为1
        control_and_target_mask = (1 << control) | (1 << target)
        
        is_list_backend = isinstance(rho, list)

        # --- 步骤 3: 核心计算逻辑 (原地修改) ---
        # 遍历密度矩阵的上三角部分（包括对角线），因为密度矩阵是厄米的
        for i in range(dim):
            for j in range(i, dim):
                # 检查 |i⟩ 基态是否需要翻转相位
                flip_i = (i & control_and_target_mask) == control_and_target_mask
                
                # 检查 |j⟩ 基态是否需要翻转相位
                flip_j = (j & control_and_target_mask) == control_and_target_mask

                # 计算总的相位因子。如果只有一个需要翻转，则为-1；如果两个都翻转或都不翻转，则为1。
                # 这等效于检查 (flip_i XOR flip_j)。
                if flip_i != flip_j:
                    factor = -1.0
                else:
                    factor = 1.0

                # 只有当需要施加非1相位时才进行乘法操作
                if factor != 1.0:
                    if is_list_backend:
                        # 对 ρ[i,j] 施加相位
                        rho[i][j] *= factor
                        # 由于 ρ 是厄米矩阵 (ρ[j,i] = ρ[i,j]*)，对 ρ[j,i] 施加共轭相位
                        if i != j:
                            rho[j][i] *= factor # factor 是实数，所以共轭不变
                    else: # CuPy/NumPy Backend
                        rho[i, j] *= factor
                        if i != j:
                            rho[j, i] *= factor

        # --- 步骤 4: 归一化 (可选但推荐) ---
        # 尽管 CZ 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self.normalize()
    def _apply_cp_kernel_dm(self, control: int, target: int, angle: float):
        """
        [内核][v1.5.15 最终正确性与健壮性修复版] 高效应用 Controlled-Phase (CP) 门。
        
        此内核直接在密度矩阵上进行原地修改，避免了构建和应用全局矩阵的开销。
        CP(angle) 门是一个对角矩阵，其作用是：当且仅当控制比特和目标比特都处于 |1⟩ 态时，
        给该计算基矢的振幅施加一个 e^(i*angle) 的相位。
        
        对于密度矩阵，演化规则为 ρ' = UρU†。由于 U 是对角的，
        演化简化为 ρ'_{ij} = U_i * ρ_{ij} * U_j†，其中 U_j† 是 U_j 的共轭。
        
        此版本直接实现了这个演化规则，使其成为一个完全自包含、独立的优化内核。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            angle (float): 施加的相位角度 (单位：弧度)。

        Raises:
            TypeError: 如果 `control`, `target` 或 `angle` 的类型不正确。
            ValueError: 如果 `control` 或 `target` 无效或相同。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(control, int):
            self._internal_logger.error(f"_apply_cp_kernel_dm: 'control' qubit index must be an integer, but got {type(control).__name__}.")
            raise TypeError("'control' qubit index must be an integer.")
        if not isinstance(target, int):
            self._internal_logger.error(f"_apply_cp_kernel_dm: 'target' qubit index must be an integer, but got {type(target).__name__}.")
            raise TypeError("'target' qubit index must be an integer.")
        if not isinstance(angle, (float, int)):
            self._internal_logger.error(f"_apply_cp_kernel_dm: 'angle' must be a numeric type, but got {type(angle).__name__}.")
            raise TypeError("'angle' must be a numeric type.")

        if not (0 <= control < self.num_qubits):
            self._internal_logger.error(f"_apply_cp_kernel_dm: Control qubit index {control} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Control qubit index {control} is out of range.")
        if not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"_apply_cp_kernel_dm: Target qubit index {target} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Target qubit index {target} is out of range.")
        
        if control == target:
            self._internal_logger.error(f"_apply_cp_kernel_dm: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits for CP gate cannot be the same.")

        # --- 步骤 2: 检查是否需要应用相位 ---
        # 如果角度是2π的整数倍，则相位为1，操作是单位矩阵，无需计算。
        if self._backend.isclose(angle % (2 * self._backend.math.pi), 0.0, atol=1e-12):
            self._internal_logger.debug("CP gate angle is a multiple of 2π, operation is identity. Skipping.")
            return

        self._internal_logger.debug(f"Applying CP(angle={angle:.4f}) kernel on control={control}, target={target}.")

        # --- 步骤 3: 初始化和准备 ---
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        
        # 创建一个位掩码，用于高效检查两个比特是否都为1
        control_and_target_mask = (1 << control) | (1 << target)
        
        # 预计算相位因子
        phase = self._backend.cmath.exp(1j * angle)
        phase_conj = phase.conjugate()
        
        is_list_backend = isinstance(rho, list)

        # --- 步骤 4: 核心计算逻辑 (原地修改) ---
        # 遍历密度矩阵的上三角部分（包括对角线），因为密度矩阵是厄米的
        for i in range(dim):
            for j in range(i, dim):
                # 检查 |i⟩ 基态是否需要施加相位
                apply_phase_to_i = (i & control_and_target_mask) == control_and_target_mask
                
                # 检查 |j⟩ 基态是否需要施加相位
                apply_phase_to_j = (j & control_and_target_mask) == control_and_target_mask

                # 计算总的相位因子 ρ'_{ij} = U_i * ρ_{ij} * U_j†
                # U_k = e^(i*angle) if k_c=1 and k_t=1, else 1.
                # U_k† = e^(-i*angle) if k_c=1 and k_t=1, else 1.
                
                factor = 1.0 + 0.0j
                if apply_phase_to_i:
                    factor *= phase
                if apply_phase_to_j:
                    factor *= phase_conj

                # 只有当需要施加非1相位时才进行乘法操作
                if not self._backend.isclose(factor, 1.0 + 0.0j):
                    if is_list_backend:
                        # 对 ρ[i,j] 施加相位
                        rho[i][j] *= factor
                        # 由于 ρ 是厄米矩阵 (ρ[j,i] = ρ[i,j]*)，对 ρ[j,i] 施加共轭相位
                        if i != j:
                            rho[j][i] *= factor.conjugate()
                    else: # CuPy/NumPy Backend
                        rho[i, j] *= factor
                        if i != j:
                            rho[j, i] *= factor.conjugate()

        # --- 步骤 5: 归一化 (可选但推荐) ---
        # 尽管 CP 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self.normalize()
    
    
    def _apply_crz_kernel_dm(self, control: int, target: int, phi: float):
        """
        [内核][v1.5.15 最终正确性与健 robuste性修复版] 高效应用 Controlled-RZ(phi) 门。
        
        此内核直接在密度矩阵上进行原地修改，避免了构建和应用全局矩阵的开销。
        CRZ(phi) 门是一个对角矩阵，其作用是：当且仅当控制比特处于 |1⟩ 态时，
        在目标比特上应用一个 RZ(phi) 旋转。
        
        对于密度矩阵，演化规则为 ρ' = UρU†。由于 U 是对角的，
        演化简化为 ρ'_{ij} = U_i * ρ_{ij} * U_j†，其中 U_j† 是 U_j 的共轭。

        此版本直接实现了这个演化规则，使其成为一个完全自包含、独立的优化内核。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            phi (float): 施加的旋转角度 (单位：弧度)。

        Raises:
            TypeError: 如果 `control`, `target` 或 `phi` 的类型不正确。
            ValueError: 如果 `control` 或 `target` 无效或相同。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(control, int):
            self._internal_logger.error(f"_apply_crz_kernel_dm: 'control' qubit index must be an integer, but got {type(control).__name__}.")
            raise TypeError("'control' qubit index must be an integer.")
        if not isinstance(target, int):
            self._internal_logger.error(f"_apply_crz_kernel_dm: 'target' qubit index must be an integer, but got {type(target).__name__}.")
            raise TypeError("'target' qubit index must be an integer.")
        if not isinstance(phi, (float, int)):
            self._internal_logger.error(f"_apply_crz_kernel_dm: 'phi' must be a numeric type, but got {type(phi).__name__}.")
            raise TypeError("'phi' must be a numeric type.")

        if not (0 <= control < self.num_qubits):
            self._internal_logger.error(f"_apply_crz_kernel_dm: Control qubit index {control} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Control qubit index {control} is out of range.")
        if not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"_apply_crz_kernel_dm: Target qubit index {target} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Target qubit index {target} is out of range.")
        
        if control == target:
            self._internal_logger.error(f"_apply_crz_kernel_dm: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits for CRZ gate cannot be the same.")

        # --- 步骤 2: 检查是否需要应用相位 ---
        # 如果角度是2π的整数倍，则RZ门是（全局相位的）单位矩阵，操作可以简化或跳过。
        if self._backend.isclose(phi % (2 * self._backend.math.pi), 0.0, atol=1e-12):
            self._internal_logger.debug("CRZ gate angle is a multiple of 2π, operation is identity (up to global phase). Skipping.")
            return

        self._internal_logger.debug(f"Applying CRZ(angle={phi:.4f}) kernel on control={control}, target={target}.")

        # --- 步骤 3: 初始化和准备 ---
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        
        # 预计算RZ门的相位因子
        half_phi = phi / 2.0
        phase_minus = self._backend.cmath.exp(-1j * half_phi) # 用于目标比特为 |0⟩
        phase_plus = self._backend.cmath.exp(1j * half_phi)   # 用于目标比特为 |1⟩
        
        control_mask = 1 << control
        target_mask = 1 << target
        
        is_list_backend = isinstance(rho, list)

        # --- 步骤 4: 核心计算逻辑 (原地修改) ---
        # 遍历密度矩阵的上三角部分（包括对角线），因为密度矩阵是厄米的
        for i in range(dim):
            for j in range(i, dim):
                # 检查控制条件是否对 |i⟩ 和 |j⟩ 基态都满足
                apply_to_i = (i & control_mask) != 0
                apply_to_j = (j & control_mask) != 0
                
                # 如果两个都不满足控制条件，则该矩阵元素不变
                if not apply_to_i and not apply_to_j:
                    continue

                # 根据目标比特的值确定相位因子
                phase_i = phase_plus if ((i & target_mask) != 0) else phase_minus
                phase_j = phase_plus if ((j & target_mask) != 0) else phase_minus
                
                # 计算总的相位因子 ρ'_{ij} = U_i * ρ_{ij} * U_j†
                # U_k = RZ_phase_k if control_k=1, else 1.
                # U_k† = RZ_phase_k* if control_k=1, else 1.
                
                factor_i = phase_i if apply_to_i else 1.0 + 0.0j
                factor_j_conj = phase_j.conjugate() if apply_to_j else 1.0 + 0.0j
                
                total_factor = factor_i * factor_j_conj

                # 只有当需要施加非1相位时才进行乘法操作
                if not self._backend.isclose(total_factor, 1.0 + 0.0j):
                    if is_list_backend:
                        # 对 ρ[i,j] 施加相位
                        rho[i][j] *= total_factor
                        # 由于 ρ 是厄米矩阵 (ρ[j,i] = ρ[i,j]*)，对 ρ[j,i] 施加共轭相位
                        if i != j:
                            rho[j][i] *= total_factor.conjugate()
                    else: # CuPy/NumPy Backend
                        rho[i, j] *= total_factor
                        if i != j:
                            rho[j, i] *= total_factor.conjugate()

        # --- 步骤 5: 归一化 (可选但推荐) ---
        # 尽管 CRZ 门是酉操作，但为了防止浮点误差累积，在每个门操作后进行归一化
        # 是一个保持数值稳定性的良好实践。
        self.normalize()
    
    
    
    def _apply_crx_kernel_dm(self, control: int, target: int, theta: float):
        """
        [内核][v1.5.15 最终正确性与健壮性修复版] 高效应用 Controlled-RX(theta) 门。
        
        此内核直接在密度矩阵上进行修改，避免了构建和应用全局矩阵的开销。
        CRX 门仅在控制比特处于 |1⟩ 态时，在目标比特上应用一个 RX(θ) 旋转。
        
        对于密度矩阵，演化规则为 ρ' = UρU†。由于 U 不再是对角的，这会混合
        密度矩阵的不同元素。此内核通过遍历所有元素并应用正确的变换规则来计算新矩阵。

        此版本是一个完全自包含、独立的优化内核，不再委托给通用旋转函数。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            theta (float): 施加的旋转角度 (单位：弧度)。

        Raises:
            TypeError: 如果 `control`, `target` 或 `theta` 的类型不正确。
            ValueError: 如果 `control` 或 `target` 无效或相同。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(control, int):
            self._internal_logger.error(f"_apply_crx_kernel_dm: 'control' qubit index must be an integer, but got {type(control).__name__}.")
            raise TypeError("'control' qubit index must be an integer.")
        if not isinstance(target, int):
            self._internal_logger.error(f"_apply_crx_kernel_dm: 'target' qubit index must be an integer, but got {type(target).__name__}.")
            raise TypeError("'target' qubit index must be an integer.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"_apply_crx_kernel_dm: 'theta' must be a numeric type, but got {type(theta).__name__}.")
            raise TypeError("'theta' must be a numeric type.")

        if not (0 <= control < self.num_qubits):
            self._internal_logger.error(f"_apply_crx_kernel_dm: Control qubit index {control} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Control qubit index {control} is out of range.")
        if not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"_apply_crx_kernel_dm: Target qubit index {target} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Target qubit index {target} is out of range.")
        
        if control == target:
            self._internal_logger.error(f"_apply_crx_kernel_dm: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits for CRX gate cannot be the same.")

        # --- 步骤 2: 检查是否需要应用旋转 ---
        # 如果角度是2π的整数倍，则RX门是单位矩阵，无需计算。
        if self._backend.isclose(theta % (2 * self._backend.math.pi), 0.0, atol=1e-12):
            self._internal_logger.debug("CRX gate angle is a multiple of 2π, operation is identity. Skipping.")
            return

        self._internal_logger.debug(f"Applying CRX(angle={theta:.4f}) kernel on control={control}, target={target}.")

        # --- 步骤 3: 初始化和准备 ---
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        
        # 预计算RX门的矩阵元素
        half_theta = theta / 2.0
        c = self._backend.math.cos(half_theta)
        s_factor = -1j * self._backend.math.sin(half_theta)
        
        control_mask = 1 << control
        target_mask = 1 << target
        
        # 创建一个新的零矩阵来存储结果，因为这是一个非对角操作，不能原地修改
        new_rho = self._backend.zeros_like(rho)
        is_list_backend = isinstance(rho, list)

        # --- 步骤 4: 核心计算逻辑 (遍历所有元素) ---
        for i in range(dim):
            # 检查行索引 i 的控制位
            is_control_i_on = (i & control_mask) != 0
            # 计算行索引 i 在目标位翻转后的索引
            i_flipped = i ^ target_mask

            for j in range(dim):
                # 检查列索引 j 的控制位
                is_control_j_on = (j & control_mask) != 0
                # 计算列索引 j 在目标位翻转后的索引
                j_flipped = j ^ target_mask
                
                # 从原始密度矩阵中提取计算新元素 ρ'[i,j] 所需的四个值
                # U_ik * ρ_kl * U†_lj = U_ik * ρ_kl * (U_jl)*
                # 如果控制位为0，U是单位矩阵。如果控制位为1，U是RX矩阵。
                
                val = 0.0 + 0.0j

                # 根据 i 和 j 的控制位，分为四种情况
                if not is_control_i_on and not is_control_j_on:
                    # U_i = I, U_j = I => ρ'_{ij} = ρ_{ij}
                    val = rho[i][j] if is_list_backend else rho[i, j]
                
                elif is_control_i_on and not is_control_j_on:
                    # U_i = RX, U_j = I => ρ'_{ij} = (c*ρ_{ij} + s*ρ_{i_f,j})
                    val = c * (rho[i][j] if is_list_backend else rho[i, j]) + \
                          s_factor * (rho[i_flipped][j] if is_list_backend else rho[i_flipped, j])

                elif not is_control_i_on and is_control_j_on:
                    # U_i = I, U_j = RX => ρ'_{ij} = (c*ρ_{ij} + s*ρ_{i,j_f})  (note: U† has s* = -s)
                    val = c * (rho[i][j] if is_list_backend else rho[i, j]) + \
                          s_factor.conjugate() * (rho[i][j_flipped] if is_list_backend else rho[i, j_flipped])

                else: # is_control_i_on and is_control_j_on
                    # U_i = RX, U_j = RX
                    val = c*c * (rho[i][j] if is_list_backend else rho[i, j]) + \
                          c*s_factor.conjugate() * (rho[i][j_flipped] if is_list_backend else rho[i, j_flipped]) + \
                          s_factor*c * (rho[i_flipped][j] if is_list_backend else rho[i_flipped, j]) + \
                          s_factor*s_factor.conjugate() * (rho[i_flipped][j_flipped] if is_list_backend else rho[i_flipped, j_flipped])

                # 将计算出的新值赋给结果矩阵
                if is_list_backend:
                    new_rho[i][j] = val
                else: # CuPy/NumPy Backend
                    new_rho[i, j] = val

        # --- 步骤 5: 更新密度矩阵并归一化 ---
        self._density_matrix = new_rho
        self.normalize()
    
    
    def _apply_cry_kernel_dm(self, control: int, target: int, theta: float):
        """
        [内核][v1.5.15 最终正确性与健 robuste性修复版] 高效应用 Controlled-RY(theta) 门。
        
        此内核直接在密度矩阵上进行修改，避免了构建和应用全局矩阵的开销。
        CRY 门仅在控制比特处于 |1⟩ 态时，在目标比特上应用一个 RY(θ) 旋转。
        
        对于密度矩阵，演化规则为 ρ' = UρU†。由于 U 不再是对角的，这会混合
        密度矩阵的不同元素。此内核通过遍历所有元素并应用正确的变换规则来计算新矩阵。

        此版本是一个完全自包含、独立的优化内核，不再委托给通用旋转函数。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            theta (float): 施加的旋转角度 (单位：弧度)。

        Raises:
            TypeError: 如果 `control`, `target` 或 `theta` 的类型不正确。
            ValueError: 如果 `control` 或 `target` 无效或相同。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(control, int):
            self._internal_logger.error(f"_apply_cry_kernel_dm: 'control' qubit index must be an integer, but got {type(control).__name__}.")
            raise TypeError("'control' qubit index must be an integer.")
        if not isinstance(target, int):
            self._internal_logger.error(f"_apply_cry_kernel_dm: 'target' qubit index must be an integer, but got {type(target).__name__}.")
            raise TypeError("'target' qubit index must be an integer.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"_apply_cry_kernel_dm: 'theta' must be a numeric type, but got {type(theta).__name__}.")
            raise TypeError("'theta' must be a numeric type.")

        if not (0 <= control < self.num_qubits):
            self._internal_logger.error(f"_apply_cry_kernel_dm: Control qubit index {control} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Control qubit index {control} is out of range.")
        if not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"_apply_cry_kernel_dm: Target qubit index {target} is out of range for {self.num_qubits} qubits.")
            raise ValueError(f"Target qubit index {target} is out of range.")
        
        if control == target:
            self._internal_logger.error(f"_apply_cry_kernel_dm: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits for CRY gate cannot be the same.")

        # --- 步骤 2: 检查是否需要应用旋转 ---
        # 如果角度是2π的整数倍，则RY门是单位矩阵，无需计算。
        if self._backend.isclose(theta % (2 * self._backend.math.pi), 0.0, atol=1e-12):
            self._internal_logger.debug("CRY gate angle is a multiple of 2π, operation is identity. Skipping.")
            return

        self._internal_logger.debug(f"Applying CRY(angle={theta:.4f}) kernel on control={control}, target={target}.")

        # --- 步骤 3: 初始化和准备 ---
        dim = 1 << self.num_qubits
        rho = self._density_matrix
        
        # 预计算RY门的矩阵元素
        half_theta = theta / 2.0
        c = self._backend.math.cos(half_theta)
        s = self._backend.math.sin(half_theta)
        
        control_mask = 1 << control
        target_mask = 1 << target
        
        # 创建一个新的零矩阵来存储结果，因为这是一个非对角操作，不能原地修改
        new_rho = self._backend.zeros_like(rho)
        is_list_backend = isinstance(rho, list)

        # --- 步骤 4: 核心计算逻辑 (遍历所有元素) ---
        for i in range(dim):
            # 检查行索引 i 的控制位
            is_control_i_on = (i & control_mask) != 0
            # 计算行索引 i 在目标位翻转后的索引
            i_flipped = i ^ target_mask

            for j in range(dim):
                # 检查列索引 j 的控制位
                is_control_j_on = (j & control_mask) != 0
                # 计算列索引 j 在目标位翻转后的索引
                j_flipped = j ^ target_mask
                
                # 从原始密度矩阵中提取计算新元素 ρ'[i,j] 所需的四个值
                # ρ'_{ij} = Σ_k,l U_ik * ρ_kl * U†_lj
                
                val = 0.0 + 0.0j

                # 根据 i 和 j 的控制位，分为四种情况
                if not is_control_i_on and not is_control_j_on:
                    # U_i = I, U_j = I => ρ'_{ij} = ρ_{ij}
                    val = rho[i][j] if is_list_backend else rho[i, j]
                
                elif is_control_i_on and not is_control_j_on:
                    # U_i = RY, U_j = I => ρ'_{ij} = c*ρ_{ij} - s*ρ_{i_f,j}
                    val = c * (rho[i][j] if is_list_backend else rho[i, j]) - \
                          s * (rho[i_flipped][j] if is_list_backend else rho[i_flipped, j])

                elif not is_control_i_on and is_control_j_on:
                    # U_i = I, U_j = RY => ρ'_{ij} = c*ρ_{ij} - s*ρ_{i,j_f} (U†_lj = U_jl)
                    val = c * (rho[i][j] if is_list_backend else rho[i, j]) - \
                          s * (rho[i][j_flipped] if is_list_backend else rho[i, j_flipped])

                else: # is_control_i_on and is_control_j_on
                    # U_i = RY, U_j = RY
                    # ρ'_{ij} = c^2*ρ_ij - c*s*ρ_i,j_f - s*c*ρ_i_f,j + s^2*ρ_i_f,j_f
                    val = c*c * (rho[i][j] if is_list_backend else rho[i, j]) - \
                          c*s * (rho[i][j_flipped] if is_list_backend else rho[i, j_flipped]) - \
                          s*c * (rho[i_flipped][j] if is_list_backend else rho[i_flipped, j]) + \
                          s*s * (rho[i_flipped][j_flipped] if is_list_backend else rho[i_flipped, j_flipped])

                # 将计算出的新值赋给结果矩阵
                if is_list_backend:
                    new_rho[i][j] = val
                else: # CuPy/NumPy Backend
                    new_rho[i, j] = val

        # --- 步骤 5: 更新密度矩阵并归一化 ---
        self._density_matrix = new_rho
        self.normalize()

    def _apply_depolarizing_channel_kernel_dm(self, target_qubits: List[int], probability: float):
        """
        [内核][v1.5.15 最终正确性与健壮性修复版] 高效应用去极化通道。
        
        此内核通过 `ρ' = (1-p)ρ + (p/3) * (XρX† + YρY† + ZρZ†)` 公式，
        在密度矩阵上应用去极化噪声。此版本修复了先前因缺少状态重置而
        导致的非对称错误，并增加了全面的输入验证和后端兼容性。

        Args:
            target_qubits (List[int]): 
                一个整数列表，指定了通道作用的目标量子比特。
            probability (float): 
                去极化错误发生的概率 `p`，必须在 [0.0, 1.0] 范围内。

        Raises:
            TypeError: 如果 `target_qubits` 或 `probability` 的类型不正确。
            ValueError: 如果 `target_qubits` 包含无效索引，或 `probability` 超出范围。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(target_qubits, list) or not all(isinstance(q, int) for q in target_qubits):
            self._internal_logger.error(f"_apply_depolarizing_channel_kernel_dm: 'target_qubits' must be a list of integers, got {type(target_qubits).__name__}.")
            raise TypeError("'target_qubits' must be a list of integers.")
        
        if not isinstance(probability, (float, int)):
            self._internal_logger.error(f"_apply_depolarizing_channel_kernel_dm: 'probability' must be a numeric type, got {type(probability).__name__}.")
            raise TypeError("'probability' must be a numeric type.")
            
        if not (0.0 <= probability <= 1.0):
            self._internal_logger.error(f"_apply_depolarizing_channel_kernel_dm: 'probability' ({probability}) must be between 0.0 and 1.0.")
            raise ValueError("'probability' must be between 0.0 and 1.0.")

        # --- 步骤 2: 检查是否需要应用噪声 ---
        # 如果错误概率非常小，则无需执行昂贵的计算。
        if self._backend.isclose(probability, 0.0, atol=1e-12):
            self._internal_logger.debug("Depolarizing probability is negligible, skipping channel application.")
            return

        p = float(probability)
        
        # --- 步骤 3: 遍历每个目标量子比特并应用通道 ---
        # 这个循环假设每个比特上的去极化噪声是独立发生的。
        for q_idx in target_qubits:
            # a) 对量子比特索引进行验证
            if not (0 <= q_idx < self.num_qubits):
                self._internal_logger.error(f"_apply_depolarizing_channel_kernel_dm: Target qubit index {q_idx} is out of range for {self.num_qubits} qubits.")
                raise ValueError(f"Target qubit index {q_idx} is out of range.")

            self._internal_logger.debug(f"Applying depolarizing channel (p={p:.4f}) on qubit {q_idx}...")
            
            # b) 安全地保存原始密度矩阵的深拷贝
            # 这是确保每个Pauli项都作用于原始状态的关键
            try:
                rho_orig = copy.deepcopy(self._density_matrix)
            except Exception as e:
                self._internal_logger.critical(f"Failed to deepcopy density matrix: {e}", exc_info=True)
                raise RuntimeError("Failed to create a copy of the density matrix for noise calculation.") from e

            # c) 计算 XρX† 部分
            self._apply_x_kernel_dm(q_idx)
            rho_x = self._density_matrix # 此刻 self._density_matrix 是 XρX†
            self._density_matrix = copy.deepcopy(rho_orig) # 重置为原始状态

            # d) 计算 YρY† 部分
            self._apply_y_kernel_dm(q_idx)
            rho_y = self._density_matrix # 此刻 self._density_matrix 是 YρY†
            
            # e) [!!! 核心修复 !!!]
            # 在计算 Z 分量之前，必须将状态重置回原始密度矩阵
            self._density_matrix = copy.deepcopy(rho_orig)
            
            # f) 计算 ZρZ† 部分
            self._apply_z_kernel_dm(q_idx)
            rho_z = self._density_matrix # 此刻 self._density_matrix 是 ZρZ†

            # g) 计算最终的线性组合: ρ' = (1-p)ρ + (p/3)(XρX† + YρY† + ZρZ†)
            # 根据后端类型（Python list 或 CuPy array）执行不同的操作
            if isinstance(rho_orig, list):
                # PurePythonBackend: 使用嵌套列表推导式进行元素级计算
                self._density_matrix = [
                    [
                        (1 - p) * r_o + (p / 3) * (r_x + r_y + r_z)
                        for r_o, r_x, r_y, r_z in zip(row_o, row_x, row_y, row_z)
                    ]
                    for row_o, row_x, row_y, row_z in zip(rho_orig, rho_x, rho_y, rho_z)
                ]
            else: 
                # CuPyBackendWrapper: 利用CuPy的向量化操作，性能更高
                self._density_matrix = (1 - p) * rho_orig + (p / 3) * (rho_x + rho_y + rho_z)
            
            # h) 归一化以消除浮点误差
            # 理论上，去极化通道是保迹的，但由于数值精度问题，最好进行归一化
            self.normalize()


    def _apply_bit_flip_channel_kernel_dm(self, target_qubits: List[int], probability: float):
        """[内核] 高效应用比特翻转通道。"""
        p = probability
        for q_idx in target_qubits:
            rho_orig = copy.deepcopy(self._density_matrix)
            self._apply_x_kernel_dm(q_idx)
            rho_x = self._density_matrix
            
            if isinstance(rho_orig, list):
                self._density_matrix = [[(1-p)*r_o + p*r_x for r_o, r_x in zip(row_o, row_x)]
                                        for row_o, row_x in zip(rho_orig, rho_x)]
            else: # CuPy
                self._density_matrix = (1-p)*rho_orig + p*rho_x
            
            self.normalize()

    def _apply_channel_kernel_dm(self, channel_type: str, target_qubits: Union[int, List[int], None], params: Dict[str, Any]):
        """[内核] 噪声通道的调度器。"""
        qubits = [target_qubits] if isinstance(target_qubits, int) else target_qubits
        if qubits is None:
            raise ValueError(f"Target qubits must be specified for channel '{channel_type}' in this kernel.")
        
        if channel_type == "depolarizing":
            self._apply_depolarizing_channel_kernel_dm(qubits, params['probability'])
        elif channel_type == "bit_flip":
            self._apply_bit_flip_channel_kernel_dm(qubits, params['probability'])
        else:
            # 对于没有内核的通道，回退到通用 Kraus 算子演化
            self._internal_logger.warning(f"No optimized kernel for channel '{channel_type}'. Using generic Kraus sum evolution.")
            self._apply_generic_channel(channel_type, qubits, params)

    def _apply_generic_channel(self, channel_type: str, target_qubits: List[int], params: Dict[str, Any]):
        """
        [后备方法 - 完整实现] 通过构建全局 Kraus 算子来应用任何量子通道。
        
        此方法是一个通用的、虽然性能较低但保证正确的后备方案。它适用于
        所有没有专门优化内核的量子通道。
        """
        log_prefix = f"[_DensityMatrixEntity._apply_generic_channel(N={self.num_qubits}, Type='{channel_type}')]"
        
        try:
            # --- 步骤 1: 获取该通道的局部 Kraus 算子列表 ---
            local_kraus_operators = self._get_kraus_operators(channel_type, params)
            if not local_kraus_operators:
                self._internal_logger.warning(f"[{log_prefix}] No Kraus operators generated for channel '{channel_type}'. No operation will be performed.")
                return

            # --- 步骤 2: 为每个目标量子比特逐一应用通道 ---
            # 这是一个简化的模型，假设每个比特上的噪声是独立的。
            # 对于多比特关联噪声，需要一次性处理所有 target_qubits。
            
            # 创建一个临时的 _StateVectorEntity 实例，只为了使用它的矩阵构建工具
            temp_builder_entity = _StateVectorEntity(self.num_qubits, self._backend)
            
            # 将 current_rho 设为初始密度矩阵
            current_rho = self._density_matrix

            for q_idx in target_qubits:
                self._internal_logger.debug(f"[{log_prefix}] Applying channel '{channel_type}' to qubit {q_idx} via generic Kraus sum...")
                
                # 初始化一个新的零矩阵，用于累加当前比特的演化结果
                new_rho_for_this_qubit = self._backend.zeros_like(current_rho)
                
                # --- 步骤 3: 遍历 Kraus 算子，计算 ρ' = Σ_i K_i ρ K_i† ---
                for local_kraus_op in local_kraus_operators:
                    
                    # a) 将局部的 2x2 Kraus 算子提升为作用于整个系统的全局算子
                    K_global = temp_builder_entity._build_global_operator_multi_qubit([q_idx], local_kraus_op)
                    
                    # b) 计算 K_i†
                    K_global_dagger = self._backend.conj_transpose(K_global)
                    
                    # c) 计算演化项 T_i = K_i * ρ * K_i†
                    term = self._backend.dot(K_global, self._backend.dot(current_rho, K_global_dagger))
                    
                    # d) 将该项累加到新的密度矩阵中
                    if isinstance(new_rho_for_this_qubit, list): # PurePythonBackend
                        for r_idx in range(len(new_rho_for_this_qubit)):
                            for c_idx in range(len(new_rho_for_this_qubit[0])):
                                new_rho_for_this_qubit[r_idx][c_idx] += term[r_idx][c_idx]
                    else: # CuPyBackendWrapper
                        new_rho_for_this_qubit += term
                
                # 将当前比特演化后的结果作为下一次循环的输入
                current_rho = new_rho_for_this_qubit

            # --- 步骤 4: 更新最终的密度矩阵 ---
            self._density_matrix = current_rho
            # 在应用完一个完整的通道后进行归一化
            self.normalize()

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] Generic channel application failed: {e}", exc_info=True)
            raise RuntimeError(f"Generic channel application failed: {e}") from e

    def _get_kraus_operators(self, channel_type: str, params: Dict[str, Any]) -> List[List[List[complex]]]:
        """
        [内部辅助方法] 根据通道类型和参数，生成其对应的局部Kraus算子列表。
        """
        kraus_ops: List[List[List[complex]]] = []
        
        # 统一从 params 字典中提取 rate/probability/gamma
        rate = params.get('probability', params.get('gamma', 0.0))
        if not isinstance(rate, (float, int)) or not (0.0 <= rate <= 1.0):
            raise ValueError(f"Invalid rate/probability/gamma parameter for channel '{channel_type}': {rate}")

        if self._backend.isclose(rate, 0.0):
            # 如果错误率为0，通道就是单位操作
            return [[[1, 0], [0, 1]]]

        # --- 根据通道类型构建 Kraus 算子 ---
        if channel_type == "depolarizing":
            sqrt_factor_I = self._backend.math.sqrt(1 - (3.0 * rate / 4.0))
            sqrt_factor_Pauli = self._backend.math.sqrt(rate / 4.0)
            kraus_ops = [
                [[sqrt_factor_I, 0], [0, sqrt_factor_I]],      # K0 = sqrt(1-3p/4)*I
                [[0, sqrt_factor_Pauli], [sqrt_factor_Pauli, 0]], # K1 = sqrt(p/4)*X
                [[0, -1j * sqrt_factor_Pauli], [1j * sqrt_factor_Pauli, 0]], # K2 = sqrt(p/4)*Y
                [[sqrt_factor_Pauli, 0], [0, -sqrt_factor_Pauli]]  # K3 = sqrt(p/4)*Z
            ]
        elif channel_type == "bit_flip":
            kraus_ops = [
                [[self._backend.math.sqrt(1 - rate), 0], [0, self._backend.math.sqrt(1 - rate)]], # K0 = sqrt(1-p)*I
                [[0, self._backend.math.sqrt(rate)], [self._backend.math.sqrt(rate), 0]]  # K1 = sqrt(p)*X
            ]
        elif channel_type == "phase_flip":
            kraus_ops = [
                [[self._backend.math.sqrt(1 - rate), 0], [0, self._backend.math.sqrt(1 - rate)]], # K0 = sqrt(1-p)*I
                [[self._backend.math.sqrt(rate), 0], [0, -self._backend.math.sqrt(rate)]] # K1 = sqrt(p)*Z
            ]
        elif channel_type == "amplitude_damping":
            kraus_ops = [
                [[1, 0], [0, self._backend.math.sqrt(1 - rate)]], # K0
                [[0, self._backend.math.sqrt(rate)], [0, 0]]      # K1
            ]
        elif channel_type == "phase_damping":
            kraus_ops = [
                [[1, 0], [0, self._backend.math.sqrt(1 - rate)]], # K0
                [[0, 0], [0, self._backend.math.sqrt(rate)]]      # K1 (注意：这是一个对角矩阵)
            ]
        else:
            # 对于任何未知的通道类型，抛出错误
            raise ValueError(f"Unknown quantum channel type for Kraus operator generation: '{channel_type}'")
        
        # 确保所有元素都是 complex 类型
        return [[[complex(v) for v in row] for row in op] for op in kraus_ops]
    def run_circuit_on_entity(self, circuit: 'QuantumCircuit'):
        """
        [最终完整版 - 基于内核映射] 在密度矩阵实体上执行线路，优先使用优化内核。
        此版本能够通过 _kernel_map 正确调度 apply_quantum_channel 指令。
        """
        log_prefix = f"[_DensityMatrixEntity.run_circuit_on_entity(N={self.num_qubits})]"
        self._internal_logger.debug(f"[{log_prefix}] Starting circuit execution on density matrix.")

        for instr_index, instruction in enumerate(circuit.instructions):
            try:
                # a) 解析指令
                gate_name = instruction[0]
                op_kwargs = instruction[-1].copy() if isinstance(instruction[-1], dict) else {}
                op_args = list(instruction[1:-1]) if isinstance(instruction[-1], dict) else list(instruction[1:])

                # b) 忽略非酉或特殊指令
                if gate_name in ('barrier', 'simulate_measurement'):
                    self._internal_logger.debug(f"[{log_prefix}] Skipping instruction '{gate_name}'.")
                    continue
                if 'condition' in op_kwargs:
                    self._internal_logger.warning(f"[{log_prefix}] Skipping conditional gate '{gate_name}'. _DensityMatrixEntity does not handle classical conditions.")
                    continue

                # --- c) 逻辑分派：优先使用内核 ---
                # 路径 1: 优化内核 (通过 _kernel_map)
                if gate_name in self._kernel_map:
                    kernel_func = self._kernel_map[gate_name]
                    self._internal_logger.debug(f"[{log_prefix}] Applying optimized density matrix kernel for '{gate_name}'.")
                    
                    # 特殊处理 apply_quantum_channel 的参数传递，因为它在kwargs中有参数
                    if gate_name == 'apply_quantum_channel':
                        kernel_func(op_args[0], op_args[1], op_kwargs.get('params', {}))
                    else:
                        kernel_func(*op_args, **op_kwargs)
                    continue # 执行成功，继续下一条指令

                # 路径 2: 通用矩阵构建 (ρ' = UρU†)
                self._internal_logger.debug(f"[{log_prefix}] Using generic matrix fallback for '{gate_name}'.")
                
                # I. 从库中获取操作定义
                op_def = QuantumOperatorLibrary.get(gate_name)
                if op_def is None:
                    # 如果内核映射中没有，库中也没有，则确实是未知门
                    raise ValueError(f"Unknown gate '{gate_name}' encountered during execution.")

                # II. 获取局部酉矩阵
                local_unitary = op_def.get_unitary(*op_args, **op_kwargs)
                if local_unitary is None:
                    raise RuntimeError(f"Could not obtain a unitary matrix representation for gate '{gate_name}'.")
                
                # III. 确定目标量子比特
                target_qubits = [q for arg in op_args for q in (arg if isinstance(arg, list) else [arg]) if isinstance(q, int)]
                unique_target_qubits = sorted(list(set(target_qubits)))

                # IV. 构建全局算子
                temp_builder_entity = _StateVectorEntity(self.num_qubits, self._backend)
                global_unitary = temp_builder_entity._build_global_operator_multi_qubit(unique_target_qubits, local_unitary)
                
                # V. 应用全局算子
                U_dagger = self._backend.conj_transpose(global_unitary)
                temp_matrix = self._backend.dot(global_unitary, self._density_matrix)
                self._density_matrix = self._backend.dot(temp_matrix, U_dagger)
                self.normalize()

            except Exception as e:
                self._internal_logger.critical(f"[{log_prefix}] Failed to execute instruction #{instr_index} ('{instruction}'): {e}", exc_info=True)
                raise RuntimeError(f"Execution of instruction '{instruction}' (index {instr_index}) failed.") from e

        self._internal_logger.debug(f"[{log_prefix}] Circuit execution on density matrix completed successfully.")


# ========================================================================
# --- 6. [核心] 量子态核心引擎 (重构为惰性求值) ---
# ========================================================================

@dataclass
class QuantumState:
    """
    [最终完整版] 一个采用惰性求值与双模式（态矢量/密度矩阵）模拟引擎的量子态容器。

    此类作为用户与量子模拟核心交互的主要接口。它将电路的“定义”与状态的“计算”分离，
    提供了灵活且高效的模拟体验。

    核心特性:
    - 惰性求值 (Statevector Mode): 在理想模拟（无噪声）模式下，应用门操作仅是将指令
      添加到内部电路中，实际的态矢量计算被推迟到需要获取结果（如概率）时才执行。
    - 双模式模拟:
      - `statevector`: 默认模式，用于高效的纯态模拟。
      - `density_matrix`: 当应用非相干噪声时，系统会自动、不可逆地切换到此模式，
        使用密度矩阵来表示混合态。
    - 实体委托: 所有实际的数值计算都被委托给内部的 `_StateVectorEntity` 或
      `_DensityMatrixEntity` 对象，这两个对象拥有专门为各自模式优化的计算内核。
    - 不可变性接口: 公共 API `run_circuit_on_state` 通过深拷贝保证了函数式
      编程的不可变性原则，即原始状态对象不会被修改。
    """
    
    # ========================================================
    # --- 1. 核心公共属性 ---
    # ========================================================

    num_qubits: int = field(
        default=0,
        metadata={'description': '此量子态包含的量子比特数量。'}
    )
    
    circuit: 'QuantumCircuit' = field(
        default=None, 
        repr=False,
        metadata={'description': '内部存储的量子线路，用于在惰性模式下收集指令。'}
    )

    # ========================================================
    # --- 2. 模式转换与历史追溯字段 (实现可回溯性) ---
    # ========================================================

    _simulation_mode: Literal['statevector', 'density_matrix'] = field(
        default='statevector', 
        repr=False,
        metadata={'description': "内部模拟模式状态机: 'statevector' (纯态) 或 'density_matrix' (混合态)。"}
    )

    mode_transition_log: Optional[Dict[str, Any]] = field(
        default=None, 
        repr=False,
        metadata={'description': '记录从纯态到混合态转换事件的详细日志，用于回溯和审计。'}
    )

    # ========================================================
    # --- 3. 内部状态与缓存 (由 __post_init__ 和内部方法管理) ---
    # ========================================================

    _density_matrix: Any = field(
        default=None, 
        repr=False, 
        init=False,
        metadata={'description': "存储计算出的密度矩阵 (仅在 'density_matrix' 模式下使用)。"}
    )
    
    _cached_statevector_entity: Optional['_StateVectorEntity'] = field(
        default=None, 
        repr=False, 
        init=False,
        metadata={'description': "缓存的态矢量实体对象，包含计算出的态矢量。"}
    )
    
    _cached_density_matrix_entity: Optional['_DensityMatrixEntity'] = field(
        default=None, 
        repr=False, 
        init=False,
        metadata={'description': "缓存的密度矩阵实体对象，提供优化的密度矩阵操作内核。"}
    )
    
    _is_cache_valid: bool = field(
        default=False, 
        repr=False, 
        init=False,
        metadata={'description': "标志位，指示缓存的实体是否与当前电路指令同步。"}
    )
    
    _backend: Any = field(
        default=None, 
        repr=False, 
        init=False,
        metadata={'description': '指向当前计算后端实例 (PurePythonBackend或CuPyBackendWrapper) 的引用。'}
    ) 
    
    _classical_registers: Dict[int, int] = field(
        default_factory=dict, 
        repr=False,
        metadata={'description': '存储经典测量结果的内部寄存器。'}
    )

    # ========================================================
    # --- 4. 元数据与日志记录 ---
    # ========================================================

    entangled_sets: List[Tuple[int, ...]] = field(
        default_factory=list,
        metadata={'description': '记录已知的纠缠量子比特集合（用于未来优化）。'}
    )
    
    gate_application_history: List[Dict[str, Any]] = field(
        default_factory=list, 
        repr=False,
        metadata={'description': '记录应用于此状态的操作历史，用于调试和审计。'}
    )
    
    measurement_outcomes_log: Dict[str, Dict[str, Any]] = field(
        default_factory=dict, 
        repr=False,
        metadata={'description': '记录详细的测量事件和结果。'}
    )
    
    system_energy_level: Optional[float] = field(
        default=0.0,
        metadata={'description': '与此状态关联的能量值（例如，在VQE中）。'}
    ) 
    
    last_significant_update_timestamp_utc_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        metadata={'description': '最后一次状态变更的UTC时间戳。'}
    )
    
    custom_state_parameters: Dict[str, Any] = field(
        default_factory=dict,
        metadata={'description': '一个用于存储用户自定义参数的字典。'}
    )

    _internal_logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(f"{__name__}.{__class__.__name__}"), 
        repr=False, 
        init=False
    )
    # --- 内部日志记录器 ---
    _internal_logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(f"{__name__}.{__class__.__name__}"), 
        repr=False, 
        init=False
    )
    # ========================================================
    # --- 5. 内部配置与类级别常量 ---
    # ========================================================

    MAX_GATE_HISTORY: int = field(
        default=50, 
        repr=False, 
        init=False,
        metadata={'description': '门操作历史记录的最大长度。'}
    )
    
    MAX_MEASUREMENT_LOG: int = field(
        default=100, 
        repr=False, 
        init=False,
        metadata={'description': '测量日志的最大长度。'}
    )
    
    SPARSITY_THRESHOLD: float = field(
        default=0.1, 
        repr=False, 
        init=False,
        metadata={'description': '未来用于稀疏态表示的阈值。'}
    )
    
    # --- 静态类型与常量 ---
    QuantumChannelType: ClassVar[Literal["depolarizing", "bit_flip", "phase_flip", "amplitude_damping", "phase_damping"]]
    _SIGMA_X_LIST: ClassVar[List[List[complex]]] = [[0+0j, 1+0j], [1+0j, 0+0j]]
    _SIGMA_Y_LIST: ClassVar[List[List[complex]]] = [[0+0j, 0-1j], [0+1j, 0+0j]]
    _SIGMA_Z_LIST: ClassVar[List[List[complex]]] = [[1+0j, 0+0j], [0+0j, -1+0j]]

    

    def __post_init__(self):
        """
        [最终完整版] 在对象初始化后，进行全面的验证、后端选择和内部状态的设置。
        """
        log_prefix = f"QuantumState.__post_init__(N={self.num_qubits})"
        self._internal_logger.debug(f"[{log_prefix}] Initializing lazy quantum state container...")

        # --- 步骤 1: 基本参数验证 ---
        if not isinstance(self.num_qubits, int) or self.num_qubits < 0:
            self._internal_logger.error(f"[{log_prefix}] Invalid 'num_qubits' '{self.num_qubits}'. Must be a non-negative integer.")
            raise ValueError("num_qubits must be a non-negative integer.")

        # --- 步骤 2: 安全地获取后端实例 ---
        try:
            self._backend = _get_backend_instance()
        except (ValueError, ImportError) as e:
            self._internal_logger.critical(f"[{log_prefix}] Failed to obtain a valid backend instance: {e}")
            raise RuntimeError("Could not initialize a computation backend for the quantum state.") from e
        
        backend_name = type(self._backend).__name__
        self._internal_logger.info(f"[{log_prefix}] Selected backend for future computations: {backend_name}")
        
        # --- 步骤 3: 检查量子比特数是否超过配置的上限 ---
        user_config_max_qubits = _core_config.get("MAX_QUBITS", 30)
        if self.num_qubits > user_config_max_qubits:
            self._internal_logger.warning(
                f"[{log_prefix}] Requested {self.num_qubits} qubits exceeds the user-configured maximum of {user_config_max_qubits}. "
                "Expanding this state may fail due to memory limits."
            )

        # --- 步骤 4: 初始化内部电路对象 ---
        if self.circuit is None:
            self.circuit = QuantumCircuit(self.num_qubits, description=f"Initial State Circuit for {self.num_qubits} qubits")
        elif not isinstance(self.circuit, QuantumCircuit):
            self._internal_logger.error(f"[{log_prefix}] If provided, 'circuit' must be a QuantumCircuit instance, but got {type(self.circuit).__name__}.")
            raise TypeError(f"If provided, 'circuit' must be a QuantumCircuit instance, but got {type(self.circuit).__name__}.")
        elif self.circuit.num_qubits != self.num_qubits:
            self._internal_logger.error(
                f"[{log_prefix}] The provided circuit's qubit number ({self.circuit.num_qubits}) does not match "
                f"the QuantumState's qubit number ({self.num_qubits})."
            )
            raise ValueError(
                f"The provided circuit's qubit number ({self.circuit.num_qubits}) does not match "
                f"the QuantumState's qubit number ({self.num_qubits})."
            )
        
        # --- 步骤 5: 初始化所有缓存状态 ---
        self._is_cache_valid = False
        self._cached_statevector_entity = None
        self._cached_density_matrix_entity = None

        # --- 步骤 6: 如果在 density_matrix 模式下初始化，则立即构建密度矩阵和实体 ---
        if self._simulation_mode == 'density_matrix':
            self._internal_logger.info(f"[{log_prefix}] Initializing directly in 'density_matrix' mode.")
            dim = 1 << self.num_qubits
            dm = self._backend.zeros((dim, dim), dtype=complex)
            if dim > 0:
                if isinstance(dm, list):
                    dm[0][0] = 1.0 + 0.0j
                else: # CuPy
                    dm[0, 0] = 1.0 + 0.0j
            self._density_matrix = dm

            # 创建并缓存密度矩阵实体
            self._cached_density_matrix_entity = _DensityMatrixEntity(self.num_qubits, self._backend, self._density_matrix)
            self.normalize()
        else:
            self._density_matrix = None

        self._internal_logger.info(f"[{log_prefix}] {self.num_qubits}-qubit lazy quantum state container created successfully in '{self._simulation_mode}' mode.")
    # ========================================================================
    # --- [新增] 哈希与指令解析辅助方法 (作为类的私有方法) ---
    # ========================================================================

    def _parse_instruction_static(self, instruction: tuple) -> Tuple[str, Tuple[Any, ...], Dict[str, Any]]:
        """一个独立的、静态行为的辅助函数，用于标准地解析一条指令元组。"""
        if not isinstance(instruction, tuple) or not instruction:
            return "", (), {}
        
        gate_name = str(instruction[0])
        has_kwargs = len(instruction) > 1 and isinstance(instruction[-1], dict)
        
        op_args = instruction[1:-1] if has_kwargs else instruction[1:]
        op_kwargs = instruction[-1] if has_kwargs else {}
        
        return gate_name, op_args, op_kwargs

    def _get_circuit_hash(self, circuit: 'QuantumCircuit') -> str:
        """
        计算 QuantumCircuit 对象的规范化哈希值。
        此版本作为 QuantumState 的一个方法，以解决作用域问题。
        """
        if not hasattr(circuit, 'num_qubits') or not hasattr(circuit, 'instructions'):
            raise TypeError("Input to _get_circuit_hash must be a QuantumCircuit-like object.")

        normalized_instructions = []
        for instr in circuit.instructions:
            # 调用本类的解析方法
            op_name, op_args, op_kwargs = self._parse_instruction_static(instr)
            
            kwargs_to_hash = {k: v for k, v in op_kwargs.items() if k not in ['name']}
            sorted_kwargs = tuple(sorted(kwargs_to_hash.items(), key=lambda item: str(item[0])))
            
            standardized_args = []
            for arg in op_args:
                if isinstance(arg, (int, float)):
                    standardized_args.append(float(arg))
                elif isinstance(arg, complex):
                    standardized_args.append((arg.real, arg.imag))
                # 增加了对 TranspiledUnitary 的处理，以提高哈希的健壮性
                elif hasattr(arg, '__class__') and arg.__class__.__name__ == 'TranspiledUnitary':
                    # 如果是 TranspiledUnitary，我们只哈希它的元数据，而不是整个矩阵
                    standardized_args.append(f"<TranspiledUnitary hash='{arg.source_circuit_hash}'>")
                elif hasattr(arg, 'tolist'):
                    list_rep = arg.tolist()
                    standardized_args.append(tuple(tuple(v) for v in list_rep))
                elif isinstance(arg, list):
                    standardized_args.append(tuple(tuple(v) for v in arg))
                else:
                    standardized_args.append(str(arg))

            normalized_instructions.append((op_name, tuple(standardized_args), sorted_kwargs))
        
        circuit_repr = (
            circuit.num_qubits,
            tuple(normalized_instructions)
        )
        
        try:
            import hashlib
            import json
            circuit_json = json.dumps(circuit_repr, sort_keys=True, default=str)
            return hashlib.sha256(circuit_json.encode('utf-8')).hexdigest()
        except Exception as e:
            self._internal_logger.error(f"Error serializing circuit for hashing: {e}. Using fallback hash.")
            import hashlib # 确保 hashlib 在此作用域内可用
            return hashlib.sha256(str(circuit_repr).encode('utf-8')).hexdigest()

    def __deepcopy__(self, memo):
        """
        [v6.1.18 - 依赖注入与序列化终极修复版] 为惰性求值的 QuantumState 类自定义深拷贝行为。
        
        此方法确保在深拷贝 QuantumState 对象时，能够正确处理其复杂的内部状态，
        特别是那些不可序列化（pickle）的属性，如后端实例和日志记录器。

        核心修复与增强:
        - 显式排除不可序列化属性: 通过一个明确的 `if field_name in (...)` 检查，
          排除了 `_backend`, `_internal_logger` 以及所有缓存实体 (`_cached_*`)
          的深拷贝尝试。这是解决 `cannot pickle 'module' object` 错误的关键。
        - 正确处理后端和日志器:
          - `_backend`: 被浅拷贝（复制引用），因为所有 QuantumState 实例在
            一次模拟中应该共享同一个计算后端。
          - `_internal_logger`: 不被复制，而是在新创建的副本上重新生成一个
            新的日志器实例。这确保了每个对象都有自己独立的日志记录器，
            避免了状态共享和潜在的序列化问题。
        - 缓存状态的正确重置: 新创建的副本的状态被明确地重置为“无缓存”
          (`_cached_* = None`, `_is_cache_valid = False`)。这是一个关键的
          逻辑步骤，确保了副本在被修改或查询时会重新进行计算，而不是
          错误地使用来自原始对象的陈旧缓存。
        - 模式感知的密度矩阵拷贝: `_density_matrix` 属性只有在当前处于
          `density_matrix` 模式下才会被深拷贝，否则在新副本中被设为 `None`，
          保持了状态的一致性。
        - 遵循 `copy` 模块的最佳实践: 正确地使用了 `memo` 字典来处理循环引用，
          并在一开始就将新创建的实例 `result` 放入 `memo` 中。
        """
        cls = self.__class__
        # 1. 创建一个新实例，但不调用 __init__
        result = cls.__new__(cls)
        # 2. 将新实例放入 memo 字典，以处理循环引用
        memo[id(self)] = result

        # 3. 遍历所有 dataclass 字段
        for f in fields(cls):
            field_name = f.name
            
            # 4. [!!! 终极修复 !!!]
            #    显式地跳过所有已知不可序列化或应特殊处理的属性
            if field_name in (
                '_backend',                      # 后端实例 (包含模块引用)
                '_internal_logger',              # 日志器实例 (可能包含不可序列化对象)
                '_cached_statevector_entity',    # 缓存的计算实体
                '_cached_density_matrix_entity', # 缓存的计算实体
                '_is_cache_valid',               # 缓存状态标志
                '_density_matrix'                # 将在下面单独处理
            ):
                continue
            
            # 5. 对于所有其他可安全拷贝的属性，执行标准的深拷贝
            value_to_copy = getattr(self, field_name)
            setattr(result, field_name, copy.deepcopy(value_to_copy, memo))

        # --- 6. 手动处理被跳过的特殊属性 ---

        # a) 后端实例：进行浅拷贝（复制引用）
        setattr(result, '_backend', self._backend)
        
        # b) 日志记录器：为新对象创建一个全新的日志器实例
        setattr(result, '_internal_logger', logging.getLogger(f"{__name__}.{cls.__name__}"))
        
        # c) 缓存状态：新副本总是从“无缓存”状态开始
        setattr(result, '_cached_statevector_entity', None)
        setattr(result, '_cached_density_matrix_entity', None)
        setattr(result, '_is_cache_valid', False)

        # d) 模式感知的密度矩阵拷贝
        if self._simulation_mode == 'density_matrix' and self._density_matrix is not None:
            # 只有在密度矩阵模式下，才深拷贝密度矩阵数据本身
            setattr(result, '_density_matrix', copy.deepcopy(self._density_matrix, memo))
        else:
            # 否则，确保新副本的密度矩阵为 None
            setattr(result, '_density_matrix', None)
        
        # e) 确保 simulation_mode 被正确复制 (因为它在上面的 for 循环中已经被深拷贝了，
        #    但为了清晰起见，可以再次确认)
        # setattr(result, '_simulation_mode', self._simulation_mode)

        # 7. 返回构建完成的深拷贝副本
        return result
    # --- [新架构] 缓存管理与展开计算核心 ---

    def _invalidate_cache(self):
        """
        [健壮性改进版] 当电路被修改时，重置所有缓存的计算结果。
        """
        if not self._is_cache_valid:
            self._internal_logger.debug(f"Cache for {self.num_qubits}-qubit QuantumState is already invalid. No action needed.")
            return

        try:
            self._internal_logger.debug(
                f"Invalidating cached state for {self.num_qubits}-qubit QuantumState. "
                "The state vector will be recomputed on the next query."
            )
            self._cached_statevector_entity = None
            self._is_cache_valid = False
            
        except Exception as e:
            self._internal_logger.critical(
                f"An unexpected error occurred during cache invalidation: {e}",
                exc_info=True
            )
            # 即使发生错误，也确保缓存状态被重置
            self._is_cache_valid = False
            self._cached_statevector_entity = None
            
    def _expand_to_statevector(self, topology: Optional[Dict[int, List[int]]] = None):
        """
        【核心-健壮性改进版】惰性求值的触发点。
        如果缓存无效，则执行内部存储的电路，生成完整的态矢量并将其封装在一个
        _StateVectorEntity 对象中进行缓存。这个函数封装了所有昂贵的计算。
        此版本接收 `topology` 参数，并将其传递给 `nexus_optimizer` 以进行拓扑感知的编译。
        """
        if self._is_cache_valid and self._cached_statevector_entity is not None:
            self._internal_logger.debug("Using valid cached state vector for computation.")
            return

        # [健壮性改进] 如果处于密度矩阵模式，不应尝试展开为态矢量
        if self._simulation_mode == 'density_matrix':
            self._internal_logger.warning(f"Attempted to _expand_to_statevector in 'density_matrix' mode. This operation is not supported. Ignoring.")
            return

        log_prefix = f"QuantumState._expand(N={self.num_qubits})"
        self._internal_logger.info(
            f"[{log_prefix}] Expanding state from a circuit with {len(self.circuit.instructions)} gates. "
            "This may take significant time and memory..."
        )
        
        start_time = time.perf_counter()

        try:
            initial_entity = _StateVectorEntity(self.num_qubits, self._backend)
            circuit_to_execute = self.circuit

            try:
                import nexus_optimizer as nopt
                
                if hasattr(nopt, 'Optimizer') and hasattr(nopt.Optimizer, 'compile'):
                    optimizer = nopt.Optimizer(backend_instance=self._backend) # 传入当前后端实例
                    
                    self._internal_logger.debug(f"[{log_prefix}] Compiling circuit with nexus_optimizer (Topology provided: {topology is not None})...")
                    
                    optimized_circuit = optimizer.compile(self.circuit, topology=topology)
                    
                    self._internal_logger.info(
                        f"[{log_prefix}] Circuit optimized. Original gates: {len(self.circuit.instructions)}, "
                        f"Optimized gates: {len(optimized_circuit.instructions)}."
                    )
                    circuit_to_execute = optimized_circuit
                else:
                    self._internal_logger.warning(
                        f"[{log_prefix}] 'nexus_optimizer' is available but does not have a compatible 'compile' method. "
                        "Proceeding with unoptimized execution."
                    )

            except (ImportError, AttributeError, NotImplementedError) as e:
                self._internal_logger.warning(
                    f"[{log_prefix}] 'nexus_optimizer' not found or is incompatible ({e}). "
                    "Falling back to unoptimized, direct execution on the state vector entity."
                )
            except Exception as e: # 捕获优化器内部可能抛出的通用异常
                self._internal_logger.error(f"[{log_prefix}] An unexpected error occurred during optimization: {e}", exc_info=True)
                # 即使优化失败，也尝试用原始电路继续执行，避免整个模拟失败
                self._internal_logger.warning(f"[{log_prefix}] Optimization failed. Proceeding with original (unoptimized) circuit.")
                circuit_to_execute = self.circuit

            final_entity = initial_entity # 在其上执行电路
            final_entity.run_circuit_on_entity(circuit_to_execute) # _StateVectorEntity.run_circuit_on_entity 不会返回新实体，而是修改自身
            
            if final_entity is None:
                raise RuntimeError("State vector entity was not computed after execution.")

            self._cached_statevector_entity = final_entity
            self._is_cache_valid = True
            
            duration = (time.perf_counter() - start_time) * 1000
            self._internal_logger.info(f"[{log_prefix}] State expansion completed successfully in {duration:.2f} ms.")

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] Failed to expand quantum state due to an error: {e}", exc_info=True)
            self._invalidate_cache() 
            raise RuntimeError("Failed to compute the full quantum state from the circuit.") from e
    
    # --- 属性与验证方法 (现在依赖展开计算) ---
    @property
    def density_matrix(self) -> Any:
        """
        [健壮性改进版] 一个只读属性，用于按需计算并返回当前状态的密度矩阵。
        """
        log_prefix = f"QuantumState.density_matrix(N={self.num_qubits})"
        
        # 如果已经处于密度矩阵模式，直接返回
        if self._simulation_mode == 'density_matrix' and self._density_matrix is not None:
            self._internal_logger.debug(f"[{log_prefix}] Already in 'density_matrix' mode, returning existing density matrix.")
            return self._density_matrix
            
        self._internal_logger.warning(
            f"[{log_prefix}] Accessing the '.density_matrix' property on a lazy QuantumState is a highly "
            "expensive operation, both in terms of computation and memory. "
            "It will trigger the full state vector expansion and a subsequent O(4^N) outer product calculation. "
            "Use with caution in performance-critical code."
        )

        try:
            self._expand_to_statevector()
            if self._cached_statevector_entity is None or self._cached_statevector_entity._state_vector is None:
                raise RuntimeError("State vector cache is unexpectedly None after attempting expansion.")
                
            vec = self._cached_statevector_entity._state_vector
            
            # 确保 vec 是后端兼容的类型，然后进行共轭
            vec_to_conjugate = self._backend._ensure_cupy_array(vec) if isinstance(self._backend, CuPyBackendWrapper) else vec
            
            if isinstance(vec_to_conjugate, list): # PurePythonBackend
                vec_conj = [v.conjugate() for v in vec_to_conjugate]
            else: # CuPyBackendWrapper
                vec_conj = vec_to_conjugate.conj()

            rho = self._backend.outer(vec_to_conjugate, vec_conj)
            
            # [健壮性改进] 如果是在 statevector 模式下计算，只是临时生成密度矩阵，不改变 _simulation_mode
            # 但如果需要切换到 density_matrix 模式，则应通过 _switch_to_density_matrix_mode 方法
            return rho

        except Exception as e:
            self._internal_logger.critical(
                f"[{log_prefix}] Failed to compute density matrix on demand: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to compute density matrix from state vector.") from e

    def is_valid(self, tolerance: float = 1e-9, check_trace_one: bool = True) -> bool:
        """
        [健壮性改进版] 对量子态进行全面的有效性检查。

        Args:
            tolerance (float, optional): 浮点数比较的容差。默认为 1e-9。
            check_trace_one (bool, optional): 是否检查迹是否为1。默认为 True。

        Returns:
            bool: 如果量子态有效，返回 `True`，否则返回 `False`。
        """
        log_prefix = f"QuantumState.is_valid(N={self.num_qubits})"

        # --- 1. 基本参数验证 ---
        if not isinstance(self.num_qubits, int) or self.num_qubits < 0:
            self._internal_logger.warning(
                f"[{log_prefix}] State validation failed: num_qubits ({self.num_qubits}) is not a non-negative integer."
            )
            return False
        if not isinstance(tolerance, (float, int)) or tolerance < 0:
            self._internal_logger.warning(f"[{log_prefix}] Invalid 'tolerance' value {tolerance}. Must be non-negative float/int. Using default 1e-9.")
            tolerance = 1e-9
        if not isinstance(check_trace_one, bool):
            self._internal_logger.warning(f"[{log_prefix}] 'check_trace_one' must be a boolean. Using default True.")
            check_trace_one = True

        user_config_max_qubits = _core_config.get("MAX_QUBITS", 30)
        if self.num_qubits > user_config_max_qubits:
            self._internal_logger.warning(
                f"[{log_prefix}] State validation failed: num_qubits ({self.num_qubits}) exceeds the user-configured "
                f"maximum of {user_config_max_qubits}. This state might not be physically representable."
            )
            # return False # 不强制失败，因为懒惰状态下可以定义大比特数
        
        if not check_trace_one:
            self._internal_logger.debug(f"[{log_prefix}] Skipping norm/trace check because 'check_trace_one' is False.")
            return True

        # --- 2. 根据模式执行验证 ---
        try:
            if self._simulation_mode == 'statevector':
                self._expand_to_statevector()
                if self._cached_statevector_entity is None or self._cached_statevector_entity._state_vector is None:
                    self._internal_logger.error(f"[{log_prefix}] Validation failed: State vector could not be computed or is None.")
                    return False
                    
                vec = self._cached_statevector_entity._state_vector
                
                if isinstance(vec, list):
                    norm_sq_values = [amp.real**2 + amp.imag**2 for amp in vec]
                    norm_sq = self._backend.sum(norm_sq_values)
                else: # CuPy
                    norm_sq = self._backend.sum(self._backend.abs(vec)**2)
                
                # [核心修复] 先用 _scalar_or_array 转换为 Python 原生类型，然后取 .real
                scalar_norm_sq = self._backend._scalar_or_array(norm_sq) if hasattr(self._backend, '_scalar_or_array') else norm_sq
                norm_sq_float = float(scalar_norm_sq.real)

                if not self._backend.isclose(norm_sq_float, 1.0, atol=tolerance):
                    self._internal_logger.warning(
                        f"[{log_prefix}] State validation failed: The norm-squared of the state vector is {float(norm_sq):.12f}, "
                        f"which is not within the tolerance ({tolerance}) of 1.0. The state is not properly normalized."
                    )
                    return False
            
            elif self._simulation_mode == 'density_matrix':
                if self._density_matrix is None:
                    self._internal_logger.error(f"[{log_prefix}] Validation failed: Density matrix is None in 'density_matrix' mode.")
                    return False
                
                trace = self._backend.trace(self._density_matrix)
                if not self._backend.isclose(float(trace), 1.0, atol=tolerance):
                    self._internal_logger.warning(
                        f"[{log_prefix}] State validation failed: The trace of the density matrix is {float(trace):.12f}, "
                        f"which is not within the tolerance ({tolerance}) of 1.0. The state is not properly normalized."
                    )
                    return False
                
                # 额外检查：密度矩阵必须是厄米的 (ρ = ρ†)
                if not self._backend.allclose(self._density_matrix, self._backend.conj_transpose(self._density_matrix), atol=tolerance):
                    self._internal_logger.warning(f"[{log_prefix}] State validation failed: Density matrix is not Hermitian (ρ != ρ†).")
                    return False

                # 额外检查：密度矩阵必须是正半定的 (所有特征值非负)
                eigenvalues = self._backend.eigvalsh(self._density_matrix)
                if self._backend.min(eigenvalues) < -tolerance: # 允许微小负值
                    self._internal_logger.warning(f"[{log_prefix}] State validation failed: Density matrix has negative eigenvalues ({self._backend.min(eigenvalues):.2e}). Not positive semi-definite.")
                    return False

            else:
                self._internal_logger.error(f"[{log_prefix}] State validation failed: Unknown simulation mode '{self._simulation_mode}'.")
                return False

        except Exception as e:
            self._internal_logger.error(f"[{log_prefix}] Validation failed due to an error during state expansion or norm/trace calculation: {e}", exc_info=True)
            return False
            
        self._internal_logger.debug(f"[{log_prefix}] State validation successful.")
        return True

    def get_probabilities(self) -> List[float]:
        """
        [v1.5.6 核心修正] 获取所有计算基的测量概率。
        
        此版本修复了因错误引用 `self.math` 而导致的 `AttributeError`，
        并确保所有数学运算都通过 `self._backend` 进行，以保持后端无关性。

        Returns:
            List[float]: 一个包含所有测量概率的 Python 列表。

        Raises:
            RuntimeError: 如果底层计算失败。
        """
        log_prefix = f"QuantumState.get_probabilities(N={self.num_qubits})"
        self._internal_logger.debug(f"[{log_prefix}] Request received to get measurement probabilities.")

        try:
            probs_raw: Any
            if self._simulation_mode == 'statevector':
                self._expand_to_statevector()
                if self._cached_statevector_entity is None or self._cached_statevector_entity._state_vector is None:
                    raise RuntimeError("State vector cache is unexpectedly None after expansion. Cannot compute probabilities.")
                    
                vec = self._cached_statevector_entity._state_vector
                
                # 针对不同后端返回的数据类型，采用不同的计算方式
                if isinstance(vec, list): # PurePythonBackend
                    probs_raw = [amp.real**2 + amp.imag**2 for amp in vec]
                else: # CuPyBackendWrapper
                    probs_raw = self._backend.abs(vec)**2
            
            elif self._simulation_mode == 'density_matrix':
                if self._density_matrix is None:
                    raise RuntimeError("Density matrix is None in 'density_matrix' mode. Cannot compute probabilities.")
                
                probs_raw_backend = self._backend.diag(self._density_matrix) # 获取对角线元素
                if isinstance(probs_raw_backend, list): # PurePythonBackend
                    # 密度矩阵的对角线元素理论上是实数，取其实部以防数值误差
                    probs_raw = [p.real for p in probs_raw_backend]
                else: # CuPy
                    probs_raw = probs_raw_backend.real # CuPy数组的 .real 属性
            else:
                raise ValueError(f"Unknown simulation mode '{self._simulation_mode}' for getting probabilities.")

            # [健壮性改进] 裁剪概率值到 [0, 1] 范围，以处理浮点误差
            clipped_probs = self._backend.clip(probs_raw, 0.0, 1.0)
            
            total_prob = self._backend.sum(clipped_probs)
            # 先用 _scalar_or_array 将后端类型（如 0-d CuPy array）转换为 Python 原生类型
            scalar_total_prob = self._backend._scalar_or_array(total_prob) if hasattr(self._backend, '_scalar_or_array') else total_prob
            
            # [核心修复] 在转换为 float 之前，先取其实部 .real
            # 概率和理论上总是实数，取实部可以安全地处理因浮点误差产生的微小虚部
            total_prob_float = float(scalar_total_prob.real)

            # --- [核心修复：使用 self._backend.isclose] ---
            if not self._backend.isclose(total_prob_float, 1.0, atol=1e-7):
                self._internal_logger.warning(
                    f"[{log_prefix}] Sum of probabilities after clipping is {total_prob_float:.12f}, which is not 1.0. "
                    "Renormalizing the probability distribution."
                )
                if total_prob_float > 1e-12: # 避免除以零
                    if isinstance(clipped_probs, list):
                        final_probabilities = [p / total_prob_float for p in clipped_probs]
                    else: # CuPy
                        final_probabilities = clipped_probs / total_prob_float
                else: # 如果总概率接近0，则所有概率都设为0
                    final_probabilities = self._backend.zeros((1 << self.num_qubits,), dtype=float)
            else:
                final_probabilities = clipped_probs
            
            # 确保最终返回的是 Python 原生列表
            if hasattr(final_probabilities, 'tolist'):
                return final_probabilities.tolist()
            return list(final_probabilities) # 使用内置的 list() 转换，更通用

        except Exception as e:
            self._internal_logger.critical(
                f"[{log_prefix}] An error occurred while getting probabilities: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to get probabilities due to an internal error.") from e
    
    
    
    
    # --- 辅助方法：在缓存的实体上应用门（用于 statevector 模式下的即时操作）---
    def _apply_gate_on_cached_entity(self, gate_name: str, op_args: List[Any], op_kwargs: Dict[str, Any]):
        """
        [健壮性改进版] 在已经展开并缓存的 `_StateVectorEntity` 上立即应用一个门操作。
        此方法仅在 `_simulation_mode == 'statevector'` 且缓存有效时使用。

        Args:
            gate_name (str): 要应用的门名称。
            op_args (List[Any]): 门的位置参数。
            op_kwargs (Dict[str, Any]): 门的关键字参数。

        Raises:
            RuntimeError: 如果 `_cached_statevector_entity` 为 None。
        """
        log_prefix = f"QuantumState._apply_gate_on_cached_entity(Op: '{gate_name}', N={self.num_qubits})"
        self._internal_logger.debug(f"[{log_prefix}] Applying gate immediately on cached state vector entity.")

        if self._cached_statevector_entity is None:
            self._internal_logger.error(f"[{log_prefix}] Cannot apply gate: _cached_statevector_entity is None. This indicates a logic error.")
            raise RuntimeError("Cannot apply gate on cached entity: _cached_statevector_entity is None.")
        
        try:
            # 创建一个只包含当前门的临时电路。
            # QuantumCircuit 的 add_gate 方法会处理宏展开。
            temp_circuit = QuantumCircuit(self.num_qubits, description=f"Single-gate immediate execution: {gate_name}")
            temp_circuit.add_gate(gate_name, *op_args, **op_kwargs)
            
            # 在实体上运行这个临时电路。
            # _StateVectorEntity.run_circuit_on_entity 负责调用内核或宏分解。
            self._cached_statevector_entity.run_circuit_on_entity(temp_circuit)
            
            # 执行成功后，清空主电路的指令，因为它们的效果已体现在 _cached_statevector_entity 中。
            self.circuit.instructions = []
            self._is_cache_valid = True 
            self._internal_logger.debug(f"[{log_prefix}] Gate '{gate_name}' applied to cached entity. Main circuit cleared, cache remains valid.")

        except Exception as e:
            self._internal_logger.error(f"[{log_prefix}] Failed to apply gate '{gate_name}' to cached entity: {e}", exc_info=True)
            self._invalidate_cache() # 强制缓存失效，以防状态损坏
            raise RuntimeError(f"Failed to apply gate '{gate_name}' to cached entity.") from e

    # --- 所有门操作现在实现了双模式 ---
    # 每个门方法会根据 self._simulation_mode 选择行为：
    # - 'statevector' 模式: 如果缓存有效，直接在 _StateVectorEntity 上应用（内核或全局矩阵）；
    #                      如果缓存无效，惰性地添加到 _QuantumCircuit。
    # - 'density_matrix' 模式: 直接调用 _build_generic_unitary_operator 获取矩阵，然后应用到密度矩阵。

    def _execute_gate_logic(self, gate_name: str, op_args: List[Any], op_kwargs: Dict[str, Any], topology: Optional[Dict[int, List[int]]] = None):
        """
        [最终完整版] 统一处理所有门操作的逻辑，根据模拟模式进行分派。
        此版本将密度矩阵模式下的执行完全委托给 _DensityMatrixEntity，
        从而能够利用其内部的优化内核。

        Args:
            gate_name (str): 门操作的名称。
            op_args (List[Any]): 门操作的位置参数。
            op_kwargs (Dict[str, Any]): 门的关键字参数。
            topology (Optional[Dict[int, List[int]]]): 可选的硬件拓扑图。
        """
        log_prefix = f"QuantumState._execute_gate_logic(Op: '{gate_name}', N={self.num_qubits})"
        
        # [健壮性改进] 参数验证
        if not isinstance(gate_name, str) or not gate_name.strip():
            self._internal_logger.error(f"[{log_prefix}] Invalid 'gate_name'. Ignoring.")
            return
        if not isinstance(op_args, list):
            self._internal_logger.error(f"[{log_prefix}] Invalid 'op_args'. Ignoring.")
            return
        if not isinstance(op_kwargs, dict):
            self._internal_logger.error(f"[{log_prefix}] Invalid 'op_kwargs'. Ignoring.")
            return

        # --- 路径 A: 密度矩阵模式 ---
        if self._simulation_mode == 'density_matrix':
            self._internal_logger.debug(f"[{log_prefix}] Executing in 'density_matrix' mode for gate '{gate_name}'.")
            
            # [核心修改：委托给 _DensityMatrixEntity]
            if self._cached_density_matrix_entity is None:
                self._internal_logger.critical(f"[{log_prefix}] Density matrix entity is not initialized in density matrix mode. This indicates a state corruption.")
                raise RuntimeError("Density matrix entity is not initialized in density matrix mode.")
            
            try:
                # 1. 创建一个只包含当前单个指令的临时电路。
                #    QuantumCircuit 的 add_gate 会处理宏展开。
                temp_circuit = QuantumCircuit(self.num_qubits, description=f"Single-op for DM entity: {gate_name}")
                temp_circuit.add_gate(gate_name, *op_args, **op_kwargs)

                # 2. 将此临时电路的执行委托给密度矩阵实体。
                #    这个实体将优先使用其内部的优化内核。
                self._cached_density_matrix_entity.run_circuit_on_entity(temp_circuit)
                
                # 3. 将实体中更新后的密度矩阵同步回 QuantumState。
                self._density_matrix = self._cached_density_matrix_entity._density_matrix
                
                # 4. 归一化以消除浮点误差。
                self.normalize()
            except Exception as e:
                self._internal_logger.error(f"[{log_prefix}] Failed to apply gate '{gate_name}' via density matrix entity: {e}", exc_info=True)
                raise RuntimeError(f"Density matrix evolution failed for gate '{gate_name}'.") from e
        
        # --- 路径 B: 态矢量模式 ---
        elif self._simulation_mode == 'statevector':
            self._internal_logger.debug(f"[{log_prefix}] Executing in 'statevector' mode for gate '{gate_name}'.")
            
            if self._is_cache_valid and self._cached_statevector_entity is not None:
                # 缓存有效：直接在实体上应用门（即时演化），利用态矢量内核。
                self._apply_gate_on_cached_entity(gate_name, op_args, op_kwargs)
            else:
                # 缓存无效：惰性地将门操作添加到电路中。
                self.circuit.add_gate(gate_name, *op_args, **op_kwargs)
                self._invalidate_cache() # 标记缓存为无效，因为电路已更改
        
        # --- 路径 C: 未知模式 ---
        else:
            self._internal_logger.critical(f"[{log_prefix}] Unknown simulation mode '{self._simulation_mode}'. Gate '{gate_name}' not applied.")
            raise ValueError(f"Unknown simulation mode: '{self._simulation_mode}'")

        # --- 通用收尾工作 ---
        # 记录操作历史并更新时间戳
        self._add_history_log(gate_name, targets=[q for q in op_args if isinstance(q, int) or isinstance(q, list)], params=op_kwargs)
        self._update_timestamp()
    
    
    
    # --- 基础单比特门 ---
    def x(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"x: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        self._execute_gate_logic("x", [qubit_index], kwargs)

    def y(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"y: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        self._execute_gate_logic("y", [qubit_index], kwargs)

    def z(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"z: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        self._execute_gate_logic("z", [qubit_index], kwargs)

    def h(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"h: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        self._execute_gate_logic("h", [qubit_index], kwargs)

    def s(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"s: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        self._execute_gate_logic("s", [qubit_index], kwargs)

    def sx(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"sx: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        self._execute_gate_logic("sx", [qubit_index], kwargs)

    def sdg(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"sdg: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        self._execute_gate_logic("sdg", [qubit_index], kwargs)

    def t_gate(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"t_gate: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        self._execute_gate_logic("t_gate", [qubit_index], kwargs)

    def tdg(self, qubit_index: int, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"tdg: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        self._execute_gate_logic("tdg", [qubit_index], kwargs)

    def rx(self, qubit_index: int, theta: float, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"rx: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"rx: 'theta' must be numeric, got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")
        self._execute_gate_logic("rx", [qubit_index, theta], kwargs)

    def ry(self, qubit_index: int, theta: float, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"ry: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"ry: 'theta' must be numeric, got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")
        self._execute_gate_logic("ry", [qubit_index, theta], kwargs)

    def rz(self, qubit_index: int, phi: float, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"rz: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        if not isinstance(phi, (float, int)):
            self._internal_logger.error(f"rz: 'phi' must be numeric, got {type(phi).__name__}.")
            raise TypeError(f"Phi must be a numeric type.")
        self._execute_gate_logic("rz", [qubit_index, phi], kwargs)

    def p_gate(self, qubit_index: int, lambda_angle: float, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"p_gate: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        if not isinstance(lambda_angle, (float, int)):
            self._internal_logger.error(f"p_gate: 'lambda_angle' must be numeric, got {type(lambda_angle).__name__}.")
            raise TypeError(f"Lambda angle must be a numeric type.")
        self._execute_gate_logic("p_gate", [qubit_index, lambda_angle], kwargs)

    def u3_gate(self, qubit_index: int, theta: float, phi: float, lambda_angle: float, **kwargs: Any):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"u3_gate: Invalid 'qubit_index' {qubit_index}.")
            raise ValueError(f"Invalid qubit index {qubit_index}.")
        if not all(isinstance(a, (float, int)) for a in [theta, phi, lambda_angle]):
            self._internal_logger.error(f"u3_gate: All angles must be numeric. Got types: ({type(theta).__name__}, {type(phi).__name__}, {type(lambda_angle).__name__}).")
            raise TypeError(f"All angles (theta, phi, lambda_angle) must be numeric.")
        self._execute_gate_logic("u3_gate", [qubit_index, theta, phi, lambda_angle], kwargs)
        
    def cnot(self, control: int, target: int, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"cnot: Invalid control/target qubits ({control}, {target}).")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"cnot: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        self._execute_gate_logic("cnot", [control, target], kwargs)

    def cz(self, control: int, target: int, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"cz: Invalid control/target qubits ({control}, {target}).")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"cz: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        self._execute_gate_logic("cz", [control, target], kwargs)

    def swap(self, qubit1: int, qubit2: int, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"swap: Invalid qubits ({qubit1}, {qubit2}).")
            raise ValueError(f"Invalid qubit index for SWAP gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"swap: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for SWAP gate cannot be the same.")
        self._execute_gate_logic("swap", [qubit1, qubit2], kwargs)

    def toffoli(self, control_1: int, control_2: int, target: int, **kwargs: Any):
        qubits = [control_1, control_2, target]
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits):
            self._internal_logger.error(f"toffoli: Invalid qubit indices {qubits}.")
            raise ValueError(f"Invalid qubit index for Toffoli gate.")
        if len(set(qubits)) != 3:
            self._internal_logger.error(f"toffoli: All qubits for Toffoli must be distinct. Got {qubits}.")
            raise ValueError("All qubits for Toffoli gate must be distinct.")
        self._execute_gate_logic("toffoli", [control_1, control_2, target], kwargs)

    def fredkin(self, control: int, target_1: int, target_2: int, **kwargs: Any):
        qubits = [control, target_1, target_2]
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits):
            self._internal_logger.error(f"fredkin: Invalid qubit indices {qubits}.")
            raise ValueError(f"Invalid qubit index for Fredkin gate.")
        if len(set(qubits)) != 3:
            self._internal_logger.error(f"fredkin: All qubits for Fredkin must be distinct. Got {qubits}.")
            raise ValueError("All qubits for Fredkin gate must be distinct.")
        self._execute_gate_logic("fredkin", [control, target_1, target_2], kwargs)

    def cp(self, control: int, target: int, angle: float, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"cp: Invalid control/target qubits ({control}, {target}).")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"cp: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        if not isinstance(angle, (float, int)):
            self._internal_logger.error(f"cp: 'angle' must be numeric, got {type(angle).__name__}.")
            raise TypeError(f"Angle must be a numeric type.")
        self._execute_gate_logic("cp", [control, target, angle], kwargs)

    def crx(self, control: int, target: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"crx: Invalid control/target qubits ({control}, {target}).")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"crx: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"crx: 'theta' must be numeric, got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")
        self._execute_gate_logic("crx", [control, target, theta], kwargs)

    def cry(self, control: int, target: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"cry: Invalid control/target qubits ({control}, {target}).")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"cry: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"cry: 'theta' must be numeric, got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")
        self._execute_gate_logic("cry", [control, target, theta], kwargs)
        
    def crz(self, control: int, target: int, phi: float, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"crz: Invalid control/target qubits ({control}, {target}).")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"crz: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        if not isinstance(phi, (float, int)):
            self._internal_logger.error(f"crz: 'phi' must be numeric, got {type(phi).__name__}.")
            raise TypeError(f"Phi must be a numeric type.")
        self._execute_gate_logic("crz", [control, target, phi], kwargs)

    def controlled_u(self, control: int, target: int, u_matrix: List[List[complex]], name: str = "CU", **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [control, target]):
            self._internal_logger.error(f"controlled_u: Invalid control/target qubits ({control}, {target}).")
            raise ValueError(f"Invalid control or target qubit index.")
        if control == target:
            self._internal_logger.error(f"controlled_u: Control qubit ({control}) cannot be the same as target qubit ({target}).")
            raise ValueError("Control and target qubits cannot be the same.")
        if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
                isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
                isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2):
            self._internal_logger.error(f"controlled_u: 'u_matrix' must be a 2x2 nested list of complex numbers. Got {type(u_matrix).__name__}.")
            raise ValueError("`u_matrix` for controlled_u must be a 2x2 nested list of complex numbers.")
        self._execute_gate_logic("controlled_u", [control, target, u_matrix], {'name': name, **kwargs})

    def rxx(self, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"rxx: Invalid qubits ({qubit1}, {qubit2}).")
            raise ValueError(f"Invalid qubit index for RXX gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"rxx: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for RXX gate cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"rxx: 'theta' must be numeric, got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")
        self._execute_gate_logic("rxx", [qubit1, qubit2, theta], kwargs)
    
    def ryy(self, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"ryy: Invalid qubits ({qubit1}, {qubit2}).")
            raise ValueError(f"Invalid qubit index for RYY gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"ryy: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for RYY gate cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"ryy: 'theta' must be numeric, got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")
        self._execute_gate_logic("ryy", [qubit1, qubit2, theta], kwargs)
        
    def rzz(self, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"rzz: Invalid qubits ({qubit1}, {qubit2}).")
            raise ValueError(f"Invalid qubit index for RZZ gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"rzz: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for RZZ gate cannot be the same.")
        if not isinstance(theta, (float, int)):
            self._internal_logger.error(f"rzz: 'theta' must be numeric, got {type(theta).__name__}.")
            raise TypeError(f"Theta must be a numeric type.")
        self._execute_gate_logic("rzz", [qubit1, qubit2, theta], kwargs)

    def mcz(self, controls: List[int], target: int, **kwargs: Any):
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"mcz: Invalid 'controls' list {controls}.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"mcz: Invalid 'target' qubit {target}.")
            raise ValueError(f"Invalid target qubit index {target}.")
        if target in controls:
            self._internal_logger.error(f"mcz: Target qubit ({target}) cannot be in controls ({controls}).")
            raise ValueError("Target qubit cannot be in controls list.")
        if len(set(controls + [target])) != len(controls) + 1:
             self._internal_logger.error(f"mcz: All qubits (controls and target) must be distinct. Got controls={controls}, target={target}.")
             raise ValueError("All qubits (controls and target) must be distinct.")
        self._execute_gate_logic("mcz", [controls, target], kwargs)

    def mcx(self, controls: List[int], target: int, **kwargs: Any):
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"mcx: Invalid 'controls' list {controls}.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"mcx: Invalid 'target' qubit {target}.")
            raise ValueError(f"Invalid target qubit index {target}.")
        if target in controls:
            self._internal_logger.error(f"mcx: Target qubit ({target}) cannot be in controls ({controls}).")
            raise ValueError("Target qubit cannot be in controls list.")
        if len(set(controls + [target])) != len(controls) + 1:
             self._internal_logger.error(f"mcx: All qubits (controls and target) must be distinct. Got controls={controls}, target={target}.")
             raise ValueError("All qubits (controls and target) must be distinct.")
        self._execute_gate_logic("mcx", [controls, target], kwargs)

    def mcp(self, controls: List[int], target: int, angle: float, **kwargs: Any):
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"mcp: Invalid 'controls' list {controls}.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"mcp: Invalid 'target' qubit {target}.")
            raise ValueError(f"Invalid target qubit index {target}.")
        if target in controls:
            self._internal_logger.error(f"mcp: Target qubit ({target}) cannot be in controls ({controls}).")
            raise ValueError("Target qubit cannot be in controls list.")
        if not isinstance(angle, (float, int)):
            self._internal_logger.error(f"mcp: 'angle' must be numeric, got {type(angle).__name__}.")
            raise TypeError(f"Angle must be a numeric type.")
        if len(set(controls + [target])) != len(controls) + 1:
             self._internal_logger.error(f"mcp: All qubits (controls and target) must be distinct. Got controls={controls}, target={target}.")
             raise ValueError("All qubits (controls and target) must be distinct.")
        self._execute_gate_logic("mcp", [controls, target, angle], kwargs)

    def mcu(self, controls: List[int], target: int, u_matrix: List[List[complex]], name: str = "MCU", **kwargs: Any):
        if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in controls):
            self._internal_logger.error(f"mcu: Invalid 'controls' list {controls}.")
            raise ValueError("Invalid 'controls' list. Must be a list of valid qubit indices.")
        if not isinstance(target, int) or not (0 <= target < self.num_qubits):
            self._internal_logger.error(f"mcu: Invalid 'target' qubit {target}.")
            raise ValueError(f"Invalid target qubit index {target}.")
        if target in controls:
            self._internal_logger.error(f"mcu: Target qubit ({target}) cannot be in controls ({controls}).")
            raise ValueError("Target qubit cannot be in controls list.")
        if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
                isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
                isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2):
            self._internal_logger.error(f"mcu: 'u_matrix' must be a 2x2 nested list of complex numbers. Got {type(u_matrix).__name__}.")
            raise ValueError("`u_matrix` for mcu must be a 2x2 nested list of complex numbers.")
        if len(set(controls + [target])) != len(controls) + 1:
             self._internal_logger.error(f"mcu: All qubits (controls and target) must be distinct. Got controls={controls}, target={target}.")
             raise ValueError("All qubits (controls and target) must be distinct.")
        self._execute_gate_logic("mcu", [controls, target, u_matrix], {'name': name, **kwargs})

    def iswap(self, qubit1: int, qubit2: int, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"iswap: Invalid qubits ({qubit1}, {qubit2}).")
            raise ValueError(f"Invalid qubit index for iSWAP gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"iswap: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for iSWAP gate cannot be the same.")
        self._execute_gate_logic("iswap", [qubit1, qubit2], kwargs)

    def ecr(self, qubit1: int, qubit2: int, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"ecr: Invalid qubits ({qubit1}, {qubit2}).")
            raise ValueError(f"Invalid qubit index for ECR gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"ecr: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for ECR gate cannot be the same.")
        self._execute_gate_logic("ecr", [qubit1, qubit2], kwargs)

    def ecrdg(self, qubit1: int, qubit2: int, **kwargs: Any):
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in [qubit1, qubit2]):
            self._internal_logger.error(f"ecrdg: Invalid qubits ({qubit1}, {qubit2}).")
            raise ValueError(f"Invalid qubit index for ECRdg gate.")
        if qubit1 == qubit2:
            self._internal_logger.error(f"ecrdg: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
            raise ValueError("Qubits for ECRdg gate cannot be the same.")
        self._execute_gate_logic("ecrdg", [qubit1, qubit2], kwargs)

    # --- 操作与测量 ---
    
    def simulate_measurement(self, 
                             qubit_to_measure: Optional[int],
                             classical_register_index: Optional[int] = None,
                             collapse_state: bool = True,
                             topology: Optional[Dict[int, List[int]]] = None) -> Optional[int]:
        """
        [v1.5.15 最终测量逻辑修复版] 模拟对指定量子比特（或所有量子比特）的测量，
        并能正确处理态坍缩和缓存状态。此版本确保了在测量所有比特时返回正确的整数结果。

        工作流程:
        1. 验证所有输入参数的有效性。
        2. 根据当前的模拟模式 (`statevector` 或 `density_matrix`)，确保状态已被
           计算出来（如果之前是惰性的，则会触发展开）。
        3. 分支处理:
           a. 如果 `qubit_to_measure` 为 `None` (测量所有比特):
              - 调用 `get_probabilities()` 获取完整的概率分布。
              - 使用后端的 `choice` 方法根据概率分布随机选择一个测量结果
                （一个 0 到 2^N-1 的整数）。
           b. 如果 `qubit_to_measure` 是一个整数 (测量单个比特):
              - 调用 `get_marginal_probabilities()` 获取该比特的 P(0) 和 P(1)。
              - 使用后端的 `choice` 方法随机选择 0 或 1 作为结果。
        4. 如果 `collapse_state` 为 `True`，则根据测量结果更新内部状态：
           - 对于态矢量，将所有与测量结果不兼容的振幅置零，然后重新归一化。
           - 对于密度矩阵，构建一个新的投影算子 |m⟩⟨m| 并更新密度矩阵，
             然后重新归一化。
        5. 如果提供了 `classical_register_index`，则将测量结果存入经典寄存器。
        6. 将测量结果作为整数返回。

        Args:
            qubit_to_measure (Optional[int]): 
                要测量的量子比特索引。如果为 `None`，则测量所有量子比特。
            classical_register_index (Optional[int], optional): 
                如果提供，测量结果将存储在此经典寄存器中。默认为 None。
            collapse_state (bool, optional): 
                测量后是否坍缩量子态。默认为 True。
            topology (Optional[Dict[int, List[int]]]): 
                可选的硬件拓扑图，用于在展开计算时进行优化。默认为 None。

        Returns:
            Optional[int]: 
                测量结果的整数表示。如果测量一个比特，返回 0 或 1。
                如果测量所有比特，返回一个 0 到 2^N-1 的整数。
                如果发生严重错误，返回 None。

        Raises:
            ValueError: 如果 `qubit_to_measure` 无效，或经典寄存器索引无效。
            RuntimeError: 如果状态无法计算或测量失败。
        """
        log_prefix = f"QuantumState.SimulateMeasurement(N={self.num_qubits}, TQ={qubit_to_measure if qubit_to_measure is not None else 'All'})"
        self._internal_logger.debug(f"[{log_prefix}] Starting measurement simulation...")

        # --- 1. 输入验证 ---
        if qubit_to_measure is not None and (not isinstance(qubit_to_measure, int) or not (0 <= qubit_to_measure < self.num_qubits)):
            self._internal_logger.error(f"[{log_prefix}] Invalid 'qubit_to_measure' index {qubit_to_measure}.")
            raise ValueError(f"Invalid target qubit index: {qubit_to_measure}")
        if classical_register_index is not None and (not isinstance(classical_register_index, int) or classical_register_index < 0):
            self._internal_logger.error(f"[{log_prefix}] Invalid 'classical_register_index' {classical_register_index}.")
            raise ValueError(f"Invalid classical register index: {classical_register_index}")
        if not isinstance(collapse_state, bool):
            self._internal_logger.error(f"[{log_prefix}] 'collapse_state' must be a boolean, got {type(collapse_state).__name__}.")
            raise TypeError(f"'collapse_state' must be a boolean.")

        try:
            # --- 2. 模式处理：确保状态已被计算 ---
            if self._simulation_mode == 'statevector':
                self._expand_to_statevector(topology=topology) # 触发展开
                if self._cached_statevector_entity is None or self._cached_statevector_entity._state_vector is None:
                    raise RuntimeError("State vector could not be computed for measurement.")
                
            elif self._simulation_mode == 'density_matrix':
                if self._density_matrix is None:
                    raise RuntimeError("Density matrix is None in 'density_matrix' mode. Cannot perform measurement.")
            else:
                raise ValueError(f"Unknown simulation mode '{self._simulation_mode}' for measurement.")

            measured_outcome_int: int # 提前声明
            
            # --- [核心修复] ---
            if qubit_to_measure is None: # 测量所有量子比特
                all_probs = self.get_probabilities()
                outcomes = list(range(1 << self.num_qubits))
                
                if not all_probs or len(all_probs) != len(outcomes):
                    self._internal_logger.error(f"[{log_prefix}] get_probabilities returned an invalid distribution. Length: {len(all_probs)}.")
                    raise RuntimeError("Invalid probability distribution for all-qubit measurement.")

                measured_outcome_int = self._backend.choice(outcomes, p=all_probs)
                self._internal_logger.info(f"[{log_prefix}] Measurement outcome for all qubits: |{measured_outcome_int}⟩ ({measured_outcome_int:0{self.num_qubits}b}b)")

                if classical_register_index is not None:
                    self._classical_registers[classical_register_index] = measured_outcome_int
                    self._internal_logger.debug(f"[{log_prefix}] Result |{measured_outcome_int}⟩ stored in CR[{classical_register_index}].")
                
                # (坍缩逻辑)
                if collapse_state and self._simulation_mode == 'statevector':
                    self._internal_logger.debug(f"[{log_prefix}] Collapsing cached state vector to outcome |{measured_outcome_int}⟩...")
                    new_vec = self._backend.zeros((1 << self.num_qubits,), dtype=complex)
                    if isinstance(new_vec, list): new_vec[measured_outcome_int] = 1.0 + 0.0j
                    else: new_vec[measured_outcome_int] = 1.0 + 0.0j
                    self._cached_statevector_entity._state_vector = new_vec
                    self.circuit.instructions = []
                    self._is_cache_valid = True
                    self._internal_logger.info(f"[{log_prefix}] Cached state vector successfully collapsed to |{measured_outcome_int}⟩.")
                elif collapse_state and self._simulation_mode == 'density_matrix':
                    self._internal_logger.debug(f"[{log_prefix}] Collapsing density matrix to outcome |{measured_outcome_int}⟩...")
                    new_dm = self._backend.zeros((1 << self.num_qubits, 1 << self.num_qubits), dtype=complex)
                    if isinstance(new_dm, list): new_dm[measured_outcome_int][measured_outcome_int] = 1.0 + 0.0j
                    else: new_dm[measured_outcome_int, measured_outcome_int] = 1.0 + 0.0j
                    self._density_matrix = new_dm
                    self.normalize()
                    self._internal_logger.info(f"[{log_prefix}] Density matrix successfully collapsed to |{measured_outcome_int}⟩.")

                # 确保返回整数结果
                return measured_outcome_int
            # --- [修复结束] ---

            else: # 测量单个量子比特
                prob_0, prob_1 = self.get_marginal_probabilities(qubit_to_measure)
                if prob_0 is None or prob_1 is None:
                    self._internal_logger.error(f"[{log_prefix}] get_marginal_probabilities returned None. Aborting measurement.")
                    return None
                measured_outcome = self._backend.choice([0, 1], p=[prob_0, prob_1])
                measured_outcome_int = int(measured_outcome)
                self._internal_logger.info(f"[{log_prefix}] Measurement outcome for qubit {qubit_to_measure}: |{measured_outcome_int}⟩")
                
                if classical_register_index is not None:
                    self._classical_registers[classical_register_index] = measured_outcome_int
                
                if collapse_state:
                    self._internal_logger.debug(f"[{log_prefix}] Collapsing state to outcome |{measured_outcome_int}⟩ on qubit {qubit_to_measure}...")
                    
                    if self._simulation_mode == 'statevector':
                        vec = self._cached_statevector_entity._state_vector
                        new_vec = self._backend.zeros((1 << self.num_qubits,), dtype=complex)
                        norm_sq = 0.0
                        mask_target_qubit = 1 << qubit_to_measure
                        
                        if isinstance(vec, list):
                            for i, amp in enumerate(vec):
                                if ((i & mask_target_qubit) >> qubit_to_measure) == measured_outcome_int:
                                    new_vec[i] = amp
                                    norm_sq += amp.real**2 + amp.imag**2
                        else: # CuPy
                            indices = self._backend.arange(len(vec))
                            compatible_indices = indices[((indices >> qubit_to_measure) & 1) == measured_outcome_int]
                            new_vec[compatible_indices] = vec[compatible_indices]
                            norm_sq = self._backend.sum(self._backend.abs(new_vec)**2)

                        if float(norm_sq) > 1e-12:
                            norm = self._backend.math.sqrt(float(norm_sq))
                            final_collapsed_vec = [amp / norm for amp in new_vec] if isinstance(new_vec, list) else new_vec / norm
                            self._cached_statevector_entity._state_vector = final_collapsed_vec
                            self.circuit.instructions = []
                            self._is_cache_valid = True
                            self._internal_logger.info(f"[{log_prefix}] Cached state vector successfully collapsed and renormalized.")
                        else:
                            self._internal_logger.error(f"[{log_prefix}] Collapse failed: Norm is zero after measurement. The state is invalid. Invalidating cache.")
                            self._invalidate_cache()
                    
                    elif self._simulation_mode == 'density_matrix':
                        rho = self._density_matrix
                        dim = 1 << self.num_qubits
                        new_rho = self._backend.zeros((dim, dim), dtype=complex)
                        
                        mask_target_qubit = 1 << qubit_to_measure

                        if isinstance(rho, list):
                            for r_idx in range(dim):
                                for c_idx in range(dim):
                                    if ((r_idx & mask_target_qubit) >> qubit_to_measure) == measured_outcome_int and \
                                       ((c_idx & mask_target_qubit) >> qubit_to_measure) == measured_outcome_int:
                                        new_rho[r_idx][c_idx] = rho[r_idx][c_idx]
                        else: # CuPy
                            indices = self._backend.arange(dim)
                            row_filter = ((indices & mask_target_qubit) >> qubit_to_measure) == measured_outcome_int
                            col_filter = row_filter
                            row_filter_2d = row_filter.reshape(-1, 1)
                            col_filter_2d = col_filter.reshape(1, -1)
                            
                            filtered_rho = rho[row_filter_2d * col_filter_2d]
                            new_rho[row_filter_2d, col_filter_2d] = filtered_rho
                            
                        self._density_matrix = new_rho
                        self.normalize()
                        self._internal_logger.info(f"[{log_prefix}] Density matrix successfully collapsed and renormalized.")
                        
                self._add_history_log("simulate_measurement", targets=[qubit_to_measure], 
                                    params={"outcome": measured_outcome_int, "collapse_state": collapse_state, "classical_register_index": classical_register_index})

                return measured_outcome_int

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] An error occurred during measurement simulation: {e}", exc_info=True)
            raise RuntimeError(f"Measurement simulation failed: {e}") from e
    
    
    
    
    def apply_quantum_channel(self, 
                            channel_type: 'QuantumChannelType', 
                            target_qubits: Union[int, List[int], None],
                            params: Dict[str, Any],
                            **kwargs: Any):
        """
        [健壮性改进版] [v1.5.10 架构修复版] 将指定的量子通道（噪声模型）作用于量子态。

        此方法是向量子态引入非相干噪声的核心接口。它的关键行为是：
        1.  在应用任何非相干噪声之前，它会通过调用 `_switch_to_density_matrix_mode`
            **强制将量子态的模拟模式不可逆地切换为 'density_matrix' 模式**。
        2.  一旦处于密度矩阵模式，它会根据 `channel_type` 构建相应的 Kraus 算子。
        3.  然后，它通过 `_DensityMatrixEntity` 的优化内核（如果可用）或通用的
            Kraus 和演化 `ρ' = Σ_i K_i ρ K_i†` 来更新密度矩阵。

        核心增强功能:
        - 架构正确性: 不再直接调用 `_StateVectorEntity` 的私有方法，而是遵循
          正确的架构模式：切换到密度矩阵模式，然后委托给 `_DensityMatrixEntity` 执行。
        - 严格的输入验证: 对 `channel_type`, `target_qubits`, `params` 的类型和
          值进行全面的验证，防止无效输入导致后续计算错误。
        - 动态目标比特解析: 能够正确处理 `target_qubits` 为 `None`（对所有比特应用）、
          单个整数或整数列表的情况。
        - 健壮的参数提取: 安全地从 `params` 字典中提取噪声参数（如 `probability`），
          并进行范围检查。
        - 模块化: 将实际的计算逻辑（Kraus 和演化）委托给 `_DensityMatrixEntity`，
          保持了此方法职责的清晰性（模式管理和参数准备）。

        Args:
            channel_type (Literal["depolarizing", ...]): 
                要应用的量子通道类型。
            target_qubits (Union[int, List[int], None]): 
                通道作用的目标量子比特。如果为 `None`，则根据通道类型决定
                （例如，去极化通道将作用于所有量子比特）。
            params (Dict[str, Any]): 
                包含通道参数的字典（例如, `{'probability': 0.01}`）。
            **kwargs: 
                任何额外的关键字参数，将被记录到历史日志中。

        Raises:
            TypeError: 如果输入参数的类型不正确。
            ValueError: 如果输入参数的值无效（例如，概率超出范围）。
            RuntimeError: 如果在模式切换或通道应用过程中发生任何错误。
        """
        log_prefix = f"QuantumState.apply_quantum_channel(N={self.num_qubits}, Type='{channel_type}')"
        self._internal_logger.debug(f"[{log_prefix}] Request to apply quantum channel.")
        
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(channel_type, str) or not channel_type.strip():
            self._internal_logger.error(f"[{log_prefix}] Invalid 'channel_type' '{channel_type}'. Must be a non-empty string.")
            raise ValueError(f"Invalid 'channel_type': '{channel_type}'.")
        
        if not isinstance(params, dict):
            self._internal_logger.error(f"[{log_prefix}] 'params' must be a dictionary, got {type(params).__name__}.")
            raise TypeError(f"'params' must be a dictionary.")

        # --- 步骤 2: 模式处理：强制切换到密度矩阵模式 ---
        if self._simulation_mode == 'statevector':
            # 调用我们设计的超级转换函数
            self._switch_to_density_matrix_mode(reason=f"Application of '{channel_type}' channel")
        
        # 确保密度矩阵实体已准备就绪
        if self._cached_density_matrix_entity is None:
            self._internal_logger.critical(f"[{log_prefix}] Cannot apply channel: density matrix entity is not initialized even after mode switch.")
            raise RuntimeError("Density matrix entity is not initialized.")

        # --- 步骤 3: 解析并验证目标量子比特 ---
        actual_target_qubits: List[int]
        if target_qubits is None:
            if channel_type in ["depolarizing"]: # 允许对所有比特应用的通道
                actual_target_qubits = list(range(self.num_qubits))
            else:
                self._internal_logger.error(f"[{log_prefix}] Channel type '{channel_type}' requires 'target_qubits' to be specified, but got None.")
                raise ValueError(f"Channel type '{channel_type}' requires 'target_qubits' to be specified.")
        elif isinstance(target_qubits, int):
            if not (0 <= target_qubits < self.num_qubits):
                self._internal_logger.error(f"[{log_prefix}] Target qubit {target_qubits} is out of range for {self.num_qubits} qubits.")
                raise ValueError(f"Target qubit {target_qubits} is out of range.")
            actual_target_qubits = [target_qubits]
        elif isinstance(target_qubits, list) and all(isinstance(q, int) and 0 <= q < self.num_qubits for q in target_qubits):
            if not target_qubits:
                self._internal_logger.error(f"[{log_prefix}] 'target_qubits' list cannot be empty for channel '{channel_type}'.")
                raise ValueError(f"'target_qubits' list cannot be empty.")
            actual_target_qubits = target_qubits
        else:
            self._internal_logger.error(f"[{log_prefix}] 'target_qubits' must be an integer, a list of integers, or None, but got {target_qubits}.")
            raise TypeError("target_qubits must be an integer, a list of integers, or None.")

        # --- 步骤 4: 委托给密度矩阵实体来执行通道应用 ---
        try:
            # 创建一个只包含当前通道指令的临时电路。
            # 这是一种清晰的、将任务委托给实体执行的方式，能够利用实体内部的
            # 内核调度逻辑（`_kernel_map`）。
            temp_channel_circuit = QuantumCircuit(self.num_qubits)
            
            # 调用 QuantumCircuit 的 apply_channel 方法，它会创建一个
            # ('apply_quantum_channel', ...) 指令元组。
            temp_channel_circuit.apply_channel(channel_type, actual_target_qubits, params, **kwargs)
            
            # 在密度矩阵实体上运行这个单指令电路。
            self._cached_density_matrix_entity.run_circuit_on_entity(temp_channel_circuit)

            # --- 步骤 5: 将更新后的密度矩阵同步回 QuantumState ---
            # 这是确保 QuantumState 自身状态与实体状态同步的关键。
            self._density_matrix = self._cached_density_matrix_entity._density_matrix
            
            # 在应用完一个可能非保迹的通道后，进行归一化。
            self.normalize()

        except Exception as e:
            self._internal_logger.error(f"[{log_prefix}] Failed to apply channel '{channel_type}' via density matrix entity: {e}", exc_info=True)
            raise RuntimeError(f"Failed to apply channel '{channel_type}': {e}") from e
        
        # --- 步骤 6: 记录历史和更新时间戳 ---
        self._add_history_log(f"QuantumChannel_{channel_type}", targets=actual_target_qubits, params=params, **kwargs)
        self._update_timestamp()
        
        self._internal_logger.info(f"[{log_prefix}] Channel '{channel_type}' successfully applied on qubits {actual_target_qubits}. Simulation is now permanently in 'density_matrix' mode.")
    # --- VQA相关方法 ---
    def get_hamiltonian_expectation(self, hamiltonian: 'Hamiltonian') -> float:
        """
        [健壮性改进版] 计算给定哈密顿量在当前量子态下的期望值。

        Args:
            hamiltonian (Hamiltonian): 一个 `List[PauliString]`，表示要计算期望值的哈密顿量。

        Returns:
            float: 哈密顿量的期望值（一个实数）。

        Raises:
            TypeError: 如果 `hamiltonian` 不是 `List[PauliString]`。
            ValueError: 如果 `PauliString` 中的量子比特索引超出范围。
            RuntimeError: 如果底层计算失败。
        """
        log_prefix = f"QuantumState.get_hamiltonian_expectation(N={self.num_qubits})"
        
        if not isinstance(hamiltonian, list) or not all(isinstance(ps, PauliString) for ps in hamiltonian):
            self._internal_logger.error(f"[{log_prefix}] 'hamiltonian' must be a list of PauliString objects, got {type(hamiltonian).__name__}.")
            raise TypeError("Hamiltonian must be a list of PauliString objects.")

        total_expectation = 0.0 + 0.0j # 使用复数累加器

        try:
            for ps_index, pauli_string in enumerate(hamiltonian):
                # 1. 验证 PauliString 中的比特索引
                for q_idx in pauli_string.pauli_map:
                    if not (0 <= q_idx < self.num_qubits):
                        self._internal_logger.error(
                            f"[{log_prefix}] PauliString '{pauli_string}' (term {ps_index}) contains qubit index {q_idx} "
                            f"which is out of range for a {self.num_qubits}-qubit state."
                        )
                        raise ValueError(f"PauliString '{pauli_string}' contains qubit index {q_idx} out of range.")
                
                # 2. 构建当前 PauliString 的全局矩阵
                global_pauli_op = self._build_pauli_string_operator(pauli_string)
                
                # 3. 根据模拟模式计算期望值
                term_exp_complex: complex
                if self._simulation_mode == 'statevector':
                    self._expand_to_statevector()
                    if self._cached_statevector_entity is None:
                        raise RuntimeError("State vector could not be computed for expectation calculation.")
                    
                    vec = self._cached_statevector_entity._state_vector
                    
                    # 确保 global_pauli_op 和 vec 是后端兼容的类型
                    global_pauli_op_backend = self._backend._ensure_cupy_array(global_pauli_op) if isinstance(self._backend, CuPyBackendWrapper) else global_pauli_op
                    vec_backend = self._backend._ensure_cupy_array(vec) if isinstance(self._backend, CuPyBackendWrapper) else vec

                    # <ψ|O|ψ> = ψ† O ψ
                    if isinstance(vec_backend, list): # PurePythonBackend
                        tmp_vec = self._backend.dot(global_pauli_op_backend, [[v] for v in vec_backend])
                        tmp_vec_flat = [row[0] for row in tmp_vec]
                        vec_conj = [v.conjugate() for v in vec_backend]
                        term_exp_complex = sum(vc * tv for vc, tv in zip(vec_conj, tmp_vec_flat))
                    else: # CuPyBackendWrapper
                        term_exp_complex = self._backend.dot(vec_backend.conj(), self._backend.dot(global_pauli_op_backend, vec_backend))
                        if hasattr(term_exp_complex, 'item'): term_exp_complex = term_exp_complex.item()
                
                else: # density_matrix 模式
                    if self._density_matrix is None:
                        raise RuntimeError("Density matrix is not available for expectation calculation in density_matrix mode.")
                    
                    global_pauli_op_backend = self._backend._ensure_cupy_array(global_pauli_op) if isinstance(self._backend, CuPyBackendWrapper) else global_pauli_op
                    dm_backend = self._backend._ensure_cupy_array(self._density_matrix) if isinstance(self._backend, CuPyBackendWrapper) else self._density_matrix

                    # Tr(Oρ)
                    product_matrix = self._backend.dot(global_pauli_op_backend, dm_backend)
                    term_exp_complex = self._backend.trace(product_matrix)

                total_expectation += (pauli_string.coefficient * term_exp_complex)
            
            # [健壮性改进] 检查总期望值的虚部
            if abs(total_expectation.imag) > 1e-7:
                self._internal_logger.warning(
                    f"[{log_prefix}] Total Hamiltonian expectation value has a significant imaginary part: {total_expectation.imag:.2e}. "
                    "Returning the real part. This might indicate numerical issues or a non-Hermitian Hamiltonian term."
                )
            return float(total_expectation.real)

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] An error occurred during Hamiltonian expectation value calculation: {e}", exc_info=True)
            raise RuntimeError(f"Hamiltonian expectation value calculation failed: {e}") from e
    def get_expectation_value(self, observable_matrix: Any) -> float:
        """
        [健壮性改进版] 计算一个可观测量在当前量子态下的期望值。

        Args:
            observable_matrix (Any): 可观测量矩阵（可以是 Python list 或 CuPy array）。

        Returns:
            float: 可观测量在当前量子态下的期望值（一个实数）。

        Raises:
            TypeError: 如果 `observable_matrix` 类型不正确。
            ValueError: 如果 `observable_matrix` 维度不匹配或不是厄米矩阵。
            RuntimeError: 如果底层计算失败。
        """
        log_prefix = f"QuantumState.get_expectation_value(N={self.num_qubits})"
        
        # --- 1. 输入验证 ---
        if not isinstance(observable_matrix, (list, self._backend._cp.ndarray if isinstance(self._backend, CuPyBackendWrapper) else type(None))):
            self._internal_logger.error(f"[{log_prefix}] 'observable_matrix' must be a list or CuPy array, got {type(observable_matrix).__name__}.")
            raise TypeError(f"'observable_matrix' must be a list or CuPy array.")

        expected_dim = 1 << self.num_qubits
        
        op_backend_type = self._backend._ensure_cupy_array(observable_matrix) if isinstance(self._backend, CuPyBackendWrapper) else observable_matrix
        
        op_shape = op_backend_type.shape if hasattr(op_backend_type, 'shape') else (len(op_backend_type), len(op_backend_type[0]))
        
        if op_shape != (expected_dim, expected_dim):
            self._internal_logger.error(
                f"[{log_prefix}] Observable matrix dimensions ({op_shape}) must match the expected state "
                f"dimensions ({(expected_dim, expected_dim)})."
            )
            raise ValueError(
                f"Observable matrix dimensions ({op_shape}) must match the expected state "
                f"dimensions ({(expected_dim, expected_dim)})."
            )
        
        # 检查是否为厄米矩阵 (O = O†)
        if not self._backend.allclose(op_backend_type, self._backend.conj_transpose(op_backend_type), atol=1e-9):
            self._internal_logger.error(f"[{log_prefix}] The provided observable matrix is not Hermitian (O != O†).")
            raise ValueError("The provided observable matrix must be Hermitian.")

        try:
            exp_val_complex: complex
            if self._simulation_mode == 'statevector':
                self._internal_logger.debug(f"[{log_prefix}] Calculating expectation value using statevector formula: <ψ|O|ψ>.")
                
                self._expand_to_statevector()
                if self._cached_statevector_entity is None:
                    raise RuntimeError("State vector could not be computed.")
                
                vec = self._cached_statevector_entity._state_vector
                vec_backend = self._backend._ensure_cupy_array(vec) if isinstance(self._backend, CuPyBackendWrapper) else vec

                # <ψ|O|ψ> = ψ† O ψ
                if isinstance(vec_backend, list): # PurePythonBackend
                    tmp_vec = self._backend.dot(op_backend_type, [[v] for v in vec_backend])
                    tmp_vec_flat = [row[0] for row in tmp_vec]
                    vec_conj = [v.conjugate() for v in vec_backend]
                    exp_val_complex = self.builtins.sum(vc * tv for vc, tv in self.builtins.zip(vec_conj, tmp_vec_flat))
                else: # CuPyBackendWrapper
                    exp_val_complex = self._backend.dot(vec_backend.conj(), self._backend.dot(op_backend_type, vec_backend))
                    if hasattr(exp_val_complex, 'item'): exp_val_complex = exp_val_complex.item()
                
            else: # self._simulation_mode == 'density_matrix'
                self._internal_logger.debug(f"[{log_prefix}] Calculating expectation value using density matrix formula: Tr(Oρ).")
                
                if self._density_matrix is None:
                    raise RuntimeError("Density matrix is not available in density_matrix mode.")
                
                dm_backend = self._backend._ensure_cupy_array(self._density_matrix) if isinstance(self._backend, CuPyBackendWrapper) else self._density_matrix

                product_matrix = self._backend.dot(op_backend_type, dm_backend)
                exp_val_complex = self._backend.trace(product_matrix)

            # [健壮性改进] 检查虚部
            if self.builtins.abs(exp_val_complex.imag) > 1e-7:
                self._internal_logger.warning(
                    f"[{log_prefix}] Expectation value has a significant imaginary part: {exp_val_complex.imag:.2e}. "
                    "This might indicate numerical issues. Returning the real part."
                )
            
            return float(exp_val_complex.real)

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] An error occurred during expectation value calculation: {e}", exc_info=True)
            raise RuntimeError(f"Expectation value calculation failed: {e}") from e

    def get_expectation_value_on_subsystem(self, reduced_density_matrix: Any, observable_matrix: Any) -> float:
        """
        [健壮性改进版] [v1.5.8 builtins fix] 计算在给定约化密度矩阵上的期望值。
        此版本修复了因错误调用 self.builtins 而导致的 AttributeError，并增强了验证逻辑。

        此方法用于计算一个可观测量在某个子系统上的期望值，其数学公式为：
        `⟨O⟩_A = Tr(O_A * ρ_A)`
        其中 `ρ_A` 是子系统的约化密度矩阵，`O_A` 是作用在该子系统上的可观测量。

        核心增强功能:
        - 错误修复: 移除了对不存在的 `self.builtins` 的调用，直接使用 Python 
          内置的 `len()` 函数或通过 `self._backend` 访问抽象函数，解决了 `AttributeError`。
        - 严格的输入验证: 在执行任何计算之前，对输入的 `reduced_density_matrix` 和 
          `observable_matrix` 进行全面的类型和形状验证。
        - 厄米性检查 (Hermiticity Check): 验证 `observable_matrix` 是否为厄米矩阵 (O = O†)，
          这是物理上可观测量必须满足的条件。如果不是，则会抛出 `ValueError`。
        - 后端无关性: 所有数值计算（矩阵转换、乘法、迹）都通过 `self._backend` 
          的抽象接口进行，确保此方法能在 PurePython 和 CuPy 后端下一致地工作。
        - 数值稳定性: 在返回最终结果之前，会检查计算出的期望值的虚部。如果虚部
          显著非零（超出浮点误差范围），会记录一条警告，因为物理上期望值必须是实数。

        Args:
            reduced_density_matrix (Any): 
                约化密度矩阵 `ρ_A`。可以是 Python 嵌套列表或 CuPy 数组。
            observable_matrix (Any): 
                作用于子系统上的可观测量矩阵 `O_A`。可以是 Python 嵌套列表或 CuPy 数组。

        Returns:
            float: 子系统上的期望值（一个实数）。

        Raises:
            TypeError: 如果输入矩阵类型不正确。
            ValueError: 如果矩阵维度不匹配或 `observable_matrix` 不是厄米矩阵。
            RuntimeError: 如果底层计算失败。
        """
        log_prefix = f"QuantumState.get_expectation_value_on_subsystem"
        
        # --- 步骤 1: 严格的输入验证 ---

        # 验证 reduced_density_matrix 的类型
        if not isinstance(reduced_density_matrix, (list, self._backend._cp.ndarray if isinstance(self._backend, CuPyBackendWrapper) else type(None))):
             self._internal_logger.error(f"[{log_prefix}] 'reduced_density_matrix' must be a list or CuPy array, got {type(reduced_density_matrix).__name__}.")
             raise TypeError(f"'{log_prefix}' 'reduced_density_matrix' must be a list or CuPy array.")
        
        # 验证 observable_matrix 的类型
        if not isinstance(observable_matrix, (list, self._backend._cp.ndarray if isinstance(self._backend, CuPyBackendWrapper) else type(None))):
             self._internal_logger.error(f"[{log_prefix}] 'observable_matrix' must be a list or CuPy array, got {type(observable_matrix).__name__}.")
             raise TypeError(f"'{log_prefix}' 'observable_matrix' must be a list or CuPy array.")

        # --- 步骤 2: 将输入转换为后端兼容的类型并检查形状 ---
        
        # 安全地将输入转换为当前后端的数据类型
        rho_backend_type = self._backend._ensure_cupy_array(reduced_density_matrix) if isinstance(self._backend, CuPyBackendWrapper) else reduced_density_matrix
        op_backend_type = self._backend._ensure_cupy_array(observable_matrix) if isinstance(self._backend, CuPyBackendWrapper) else observable_matrix

        # [核心修复]
        # 错误代码: rho_shape = rho_backend_type.shape if hasattr(rho_backend_type, 'shape') else (self.builtins.len(rho_backend_type), self.builtins.len(rho_backend_type[0]))
        # 修正后的代码: 直接使用内置的 len()
        rho_shape = rho_backend_type.shape if hasattr(rho_backend_type, 'shape') else (len(rho_backend_type), len(rho_backend_type[0]) if rho_backend_type else (0,0))
        op_shape = op_backend_type.shape if hasattr(op_backend_type, 'shape') else (len(op_backend_type), len(op_backend_type[0]) if op_backend_type else (0,0))

        # 验证形状是否匹配
        if rho_shape != op_shape:
            self._internal_logger.error(f"[{log_prefix}] Reduced density matrix dimensions ({rho_shape}) and observable matrix dimensions ({op_shape}) must match.")
            raise ValueError("Reduced density matrix and observable matrix dimensions must match.")
        
        # 验证可观测量是否为厄米矩阵 (O = O†)
        if not self._backend.allclose(op_backend_type, self._backend.conj_transpose(op_backend_type), atol=1e-9):
            self._internal_logger.error(f"[{log_prefix}] The provided observable matrix is not Hermitian (O != O†).")
            raise ValueError("The provided observable matrix must be Hermitian.")

        # --- 步骤 3: 计算期望值 Tr(Oρ) ---
        try:
            # 使用后端无关的接口执行矩阵乘法和迹运算
            product_matrix = self._backend.dot(op_backend_type, rho_backend_type)
            exp_val_complex = self._backend.trace(product_matrix)

            # [核心修复]
            # 错误代码: if self.builtins.abs(exp_val_complex.imag) > 1e-7:
            # 修正后的代码: 直接使用内置的 abs()
            if abs(exp_val_complex.imag) > 1e-7:
                self._internal_logger.warning(
                    f"[{log_prefix}] Expectation value has a significant imaginary part: {exp_val_complex.imag:.2e}. "
                    "Returning the real part. This might indicate numerical issues."
                )
            
            # --- 步骤 4: 返回实部作为最终结果 ---
            return float(exp_val_complex.real)

        except Exception as e:
            # 捕获所有可能的底层计算错误
            self._internal_logger.critical(f"[{log_prefix}] An error occurred during subsystem expectation value calculation: {e}", exc_info=True)
            raise RuntimeError(f"Subsystem expectation value calculation failed: {e}") from e
    
    def _build_pauli_string_operator(self, pauli_string: 'PauliString') -> Any:
        """
        [健壮性改进版] 将一个 `PauliString` 转换为其对应的全局算子矩阵。

        Args:
            pauli_string (PauliString): PauliString 对象。

        Returns:
            Any: 全局 Pauli 算子矩阵。

        Raises:
            TypeError: 如果 `pauli_string` 不是 `PauliString` 类型。
            ValueError: 如果 `pauli_string` 包含未知 Pauli 算子。
        """
        log_prefix = f"[_StateVectorEntity._build_pauli_string_operator(N={self.num_qubits})]"
        
        # 修正: 直接使用 PauliString 类名
        if not isinstance(pauli_string, PauliString):
            self._internal_logger.error(f"[{log_prefix}] 'pauli_string' must be a PauliString instance, got {type(pauli_string).__name__}.")
            raise TypeError("'pauli_string' must be a PauliString instance.")

        pauli_ops = {
            'I': self._backend.eye(2, dtype=complex),
            'X': self._SIGMA_X_LIST,
            'Y': self._SIGMA_Y_LIST,
            'Z': self._SIGMA_Z_LIST,
        }
        
        # 如果 num_qubits 为 0，直接返回 1x1 单位矩阵
        if self.num_qubits == 0:
            return self._backend.eye(1, dtype=complex)

        # 初始化一个包含所有量子比特的单位算子列表
        # 修正: 使用 self.builtins.range
        ops_for_qubits: List[Any] = [pauli_ops['I'] for _ in self._backend.builtins.range(self.num_qubits)]
        
        for q_idx, op_char in pauli_string.pauli_map.items():
            if not (0 <= q_idx < self.num_qubits):
                self._internal_logger.error(f"[{log_prefix}] PauliString '{pauli_string}' contains qubit index {q_idx} which is out of range for a {self.num_qubits}-qubit state.")
                raise ValueError(f"PauliString '{pauli_string}' contains qubit index {q_idx} out of range.")
            
            if op_char not in pauli_ops:
                self._internal_logger.error(f"[{log_prefix}] Unknown Pauli operator '{op_char}' in PauliString: {pauli_string}.")
                raise ValueError(f"Unknown Pauli operator '{op_char}' in PauliString: {pauli_string}")
            ops_for_qubits[q_idx] = pauli_ops[op_char]
        
        # 按照约定，从最低位比特 (q0) 开始张量积
        global_op = ops_for_qubits[0]
        # 修正: 使用 self.builtins.range
        for i in self._backend.builtins.range(1, self.num_qubits):
            global_op = self._backend.kron(ops_for_qubits[i], global_op)
        
        return global_op
    def _generate_einsum_string_for_statevector_partial_trace(self, num_total_qubits: int, qubits_to_trace: List[int]) -> str:
        """
        [健壮性改进版] 为态矢量的部分迹生成 `einsum` 下标字符串。
        """
        if not self._backend.builtins.isinstance(num_total_qubits, self._backend.builtins.int) or not (0 <= num_total_qubits <= 26):
            self._internal_logger.error(f"_generate_einsum_string_for_statevector_partial_trace: Invalid 'num_total_qubits' ({num_total_qubits}). Must be between 0 and 26.")
            raise ValueError(
                f"Number of qubits ({num_total_qubits}) for einsum string generation "
                "must be between 0 and 26 for distinct alphabetical indices."
            )
        if not self._backend.builtins.isinstance(qubits_to_trace, self._backend.builtins.list) or not self._backend.builtins.all(self._backend.builtins.isinstance(q, self._backend.builtins.int) and 0 <= q < num_total_qubits for q in qubits_to_trace):
            self._internal_logger.error(f"_generate_einsum_string_for_statevector_partial_trace: Invalid 'qubits_to_trace' list {qubits_to_trace}.")
            raise ValueError("Invalid 'qubits_to_trace' list.")

        # 定义 bra 和 ket 的原始索引字符 (小写用于bra，大写用于ket，但这里我们先统一为小写)
        all_qubit_chars = [self._backend.builtins.chr(self._backend.builtins.ord('a') + i) for i in self._backend.builtins.range(num_total_qubits)]
        
        # input1 对应原始态矢量 |ψ⟩，其下标是所有量子比特的索引
        input1 = "".join(all_qubit_chars)
        
        # input2 对应共轭态矢量 ⟨ψ|，其下标根据迹掉和保留的比特来构造
        # 对于保留的比特，使用不同的大写字母（以避免与bra冲突）
        # 对于迹掉的比特，使用与bra相同的字母（因为它们会被求和）
        input2_parts: List[str] = []
        output_parts: List[str] = [] # 输出下标字符
        
        # 为了确保确定性，按照量子比特的物理索引从高到低遍历
        for i in self._backend.builtins.range(num_total_qubits -1, -1, -1): # 从 qN-1 到 q0
            # einsum 字符串是 `q_N-1 q_N-2 ... q_0` 的顺序
            # 所以这里我们使用 all_qubit_chars[i] 来表示第 i 个字符，它对应着 `q_N-1-i` 这个物理比特
            # 这里的索引 `q_string_idx = num_total_qubits - 1 - global_qubit_idx`
            
            # 我们需要的是基于物理比特索引 (0,1,2...) 来判断
            global_qubit_idx = i # 真实的量子比特索引
            einsum_char_for_this_qubit = all_qubit_chars[num_total_qubits - 1 - global_qubit_idx] # 这个比特在einsum串中的字符

            if global_qubit_idx in qubits_to_trace:
                # 如果是迹掉的比特，bra 和 ket 使用相同的字符
                input2_parts.append(einsum_char_for_this_qubit)
            else:
                # 如果是保留的比特，bra 使用小写，ket 使用大写
                input2_parts.append(einsum_char_for_this_qubit.upper())
                output_parts.append(einsum_char_for_this_qubit)
                output_parts.append(einsum_char_for_this_qubit.upper())

        # input2_parts 逆序构建，因为我们是从高位比特开始
        input2_final = "".join(self._backend.builtins.reversed(input2_parts))
        output_final = "".join(self._backend.builtins.reversed(output_parts))
        
        return f"{input1},{input2_final}->{output_final}"
    def _partial_trace(self, qubits_to_trace_out: List[int]) -> Any:
        """
        [健壮性与逻辑修复版] 对状态执行部分迹（Partial Trace）操作。
        此版本不再依赖复杂的 einsum 字符串，而是根据部分迹的数学定义直接计算，
        提高了代码的清晰度和正确性。

        Args:
            qubits_to_trace_out (List[int]): 要从量子态中迹掉的量子比特索引列表。

        Returns:
            Any: 约化后的密度矩阵（Python列表或CuPy数组）。

        Raises:
            TypeError: 如果 `qubits_to_trace_out` 类型不正确。
            ValueError: 如果 `qubits_to_trace_out` 包含无效索引或重复索引。
            RuntimeError: 如果底层计算失败。
        """
        log_prefix = f"QuantumState._partial_trace(N={self.num_qubits})"
        
        # --- 1. 输入验证 ---
        if not isinstance(qubits_to_trace_out, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits_to_trace_out):
            self._internal_logger.error(f"[{log_prefix}] 'qubits_to_trace_out' must be a list of valid qubit indices, got {qubits_to_trace_out}.")
            raise TypeError("Invalid 'qubits_to_trace_out' list.")
        
        unique_qubits_to_trace = sorted(list(set(qubits_to_trace_out)))
        if len(unique_qubits_to_trace) != len(qubits_to_trace_out):
            self._internal_logger.warning(f"[{log_prefix}] 'qubits_to_trace_out' contains duplicate indices. Using unique sorted list: {unique_qubits_to_trace}.")
            qubits_to_trace_out = unique_qubits_to_trace

        # --- 2. 模式处理和状态准备 ---
        if self._simulation_mode == 'statevector':
            self._expand_to_statevector()
            if self._cached_statevector_entity is None or self._cached_statevector_entity._state_vector is None:
                raise RuntimeError("State vector not available for partial trace.")
            
            # 从态矢量 |ψ⟩ 计算完整的密度矩阵 ρ = |ψ⟩⟨ψ|
            vec = self._cached_statevector_entity._state_vector
            vec_conj = vec.conj() if hasattr(vec, 'conj') else [v.conjugate() for v in vec]
            full_rho = self._backend.outer(vec, vec_conj)
        
        elif self._simulation_mode == 'density_matrix':
            if self._density_matrix is None:
                raise RuntimeError("Density matrix is None in 'density_matrix' mode.")
            full_rho = self._density_matrix
        else:
            raise ValueError(f"Unknown simulation mode '{self._simulation_mode}' for partial trace.")

        # --- 3. 计算部分迹的核心逻辑 ---
        try:
            qubits_to_keep = sorted([q for q in range(self.num_qubits) if q not in qubits_to_trace_out])
            num_qubits_kept = len(qubits_to_keep)
            dim_out = 1 << num_qubits_kept

            reduced_rho = self._backend.zeros((dim_out, dim_out), dtype=complex)
            
            # 遍历输出的约化密度矩阵的每个元素
            for i_out in range(dim_out):  # 对应 |i_out⟩ (行)
                for j_out in range(dim_out):  # 对应 ⟨j_out| (列)
                    sum_val = 0.0 + 0.0j

                    # 遍历所有被迹掉的比特的基态组合
                    for k_trace in range(1 << len(qubits_to_trace_out)):
                        # 将保留比特的索引 (i_out, j_out) 和迹掉比特的索引 (k_trace)
                        # 组合成完整的系统索引 (global_row_idx, global_col_idx)
                        global_row_idx = 0
                        global_col_idx = 0

                        # 添加保留比特的贡献
                        for bit_pos_kept, q_kept in enumerate(qubits_to_keep):
                            if (i_out >> bit_pos_kept) & 1:
                                global_row_idx |= (1 << q_kept)
                            if (j_out >> bit_pos_kept) & 1:
                                global_col_idx |= (1 << q_kept)
                        
                        # 添加迹掉比特的贡献
                        for bit_pos_traced, q_traced in enumerate(qubits_to_trace_out):
                            if (k_trace >> bit_pos_traced) & 1:
                                trace_mask = (1 << q_traced)
                                global_row_idx |= trace_mask
                                global_col_idx |= trace_mask # 对角线元素
                        
                        # 从完整密度矩阵中提取元素并累加
                        # ρ_A[i,j] = Σ_k ρ[ik, jk]
                        if isinstance(full_rho, list):
                            sum_val += full_rho[global_row_idx][global_col_idx]
                        else: # CuPy
                            sum_val += full_rho[global_row_idx, global_col_idx]

                    if isinstance(reduced_rho, list):
                        reduced_rho[i_out][j_out] = sum_val
                    else: # CuPy
                        reduced_rho[i_out, j_out] = sum_val
                        
            return reduced_rho

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] Partial trace failed during calculation: {e}", exc_info=True)
            raise RuntimeError(f"Partial trace calculation failed: {e}") from e
    def normalize(self):
        """
        [健壮性与正确性增强版] 对当前状态表示进行归一化。

        此方法根据当前的 `_simulation_mode` 采取不同的行为：

        1.  **`density_matrix` 模式**:
            - 计算密度矩阵 `ρ` 的迹 `Tr(ρ)`。
            - 如果迹不为 1 (在一定容差范围内)，则将密度矩阵的每个元素除以迹，
              即 `ρ' = ρ / Tr(ρ)`，以确保 `Tr(ρ') = 1`。
            - 此版本修复了因错误地将复数迹直接转换为浮点数而导致的 `TypeError`。

        2.  **`statevector` 模式**:
            - 在此模式下，`QuantumState` 对象本身不直接持有态矢量，而是持有一个
              指令列表（`self.circuit`）。
            - 实际的态矢量存在于内部的 `_StateVectorEntity` 对象中，并且该实体
              在每次应用门操作后都会**自动进行归一化**。
            - 因此，在 `statevector` 模式下调用此方法是一个**逻辑上的空操作 (no-op)**，
              因为它所代表的最终状态在计算时总是会被归一化。这样做可以避免不必要的、
              昂贵的态矢量展开计算。

        核心增强功能:
        - 修复了将复数迹转换为浮点数时发生的 `TypeError`。
        - 增加了对迹为零或接近零的检查，防止因除以零而崩溃。
        - 明确了在 `statevector` 模式下的行为，提高了代码的逻辑清晰度和性能。
        - 增加了详尽的文档和内部注释。

        Raises:
            ValueError: 如果 `_simulation_mode` 是一个未知的模式。
            RuntimeError: 如果在计算或归一化过程中发生任何未预期的底层错误。
        """
        log_prefix = f"QuantumState.normalize(Mode: {self._simulation_mode})"

        # 对于 0 比特系统，状态是固定的 ([1])，无需归一化。
        if self.num_qubits == 0:
            self._internal_logger.debug(f"[{log_prefix}] 0-qubit system. Normalization is trivial and skipped.")
            return

        try:
            # --- 路径 A: 密度矩阵模式 ---
            if self._simulation_mode == 'density_matrix':
                # 检查密度矩阵是否存在
                if self._density_matrix is None:
                    self._internal_logger.warning(f"[{log_prefix}] Called on a None density matrix in 'density_matrix' mode. Skipping normalization.")
                    return
                
                # 使用后端计算迹
                trace = self._backend.trace(self._density_matrix)
                
                # [BUGFIX] 正确地从复数迹中提取实部。
                # 理论上，密度矩阵的迹是实数，但由于浮点误差，它可能有一个微小的虚部。
                trace_float = float(trace.real)
                
                # 检查迹是否接近于零，以避免除以零的错误。
                if self._backend.isclose(trace_float, 0.0, atol=1e-12):
                    self._internal_logger.critical(
                        f"[{log_prefix}] Cannot normalize density matrix because its trace is close to zero ({trace_float:.2e}). "
                        "The quantum state may be corrupted. Normalization skipped."
                    )
                    return
                    
                # 只有当迹不接近 1 时才执行归一化，以节省计算。
                if not self._backend.isclose(trace_float, 1.0, atol=1e-9):
                    self._internal_logger.debug(f"[{log_prefix}] Normalizing density matrix with trace {trace_float:.12f}")
                    
                    # 无论是 PurePython (list) 还是 CuPy (ndarray)，
                    # 都可以通过 `/=` 或 `/` 操作符进行除法。
                    if isinstance(self._density_matrix, list): # PurePythonBackend
                        self._density_matrix = [[element / trace for element in row] for row in self._density_matrix]
                    else: # CuPyBackendWrapper
                        self._density_matrix /= trace
                else:
                    self._internal_logger.debug(f"[{log_prefix}] Density matrix is already normalized (trace is {trace_float:.12f}, within tolerance).")
            
            # --- 路径 B: 态矢量模式 ---
            elif self._simulation_mode == 'statevector':
                # 在此模式下，归一化由 _StateVectorEntity 在其内部方法（如门操作内核）
                # 调用后自动处理。外部调用此方法不应触发昂贵的态矢量展开。
                self._internal_logger.debug(
                    f"[{log_prefix}] Normalization in 'statevector' mode is handled by the internal _StateVectorEntity "
                    "after each operation. This call is a logical no-op and does not trigger state expansion."
                )
                # 不执行任何操作。
            
            # --- 路径 C: 未知模式 (错误情况) ---
            else:
                self._internal_logger.critical(f"[{log_prefix}] Unknown simulation mode encountered: '{self._simulation_mode}'. Cannot normalize.")
                raise ValueError(f"Unknown simulation mode encountered in normalize: '{self._simulation_mode}'")
        
        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] An error occurred during normalization: {e}", exc_info=True)
            raise RuntimeError(f"Normalization failed: {e}") from e
    def _generate_einsum_string_for_density_matrix_partial_trace(self, num_total_qubits: int, qubits_to_trace: List[int]) -> str:
        """
        [健壮性改进版] 为密度矩阵的部分迹生成 `einsum` 下标字符串。

        此方法构建一个 `einsum` 字符串，用于描述如何通过张量缩并来计算
        一个密度矩阵 `ρ` 的部分迹。例如，对于一个 3 比特系统，要迹掉
        比特 q1，生成的字符串可能是 `"abcABC->acAC"`。

        核心增强功能:
        - 错误修复: 移除了对不存在的 `self.builtins` 的调用，直接使用 Python 
          内置的 `isinstance`, `chr`, `ord`, `range` 等函数，解决了 `AttributeError`。
        - 严格的输入验证: 在生成字符串之前，对 `num_total_qubits` 和 
          `qubits_to_trace` 的类型和值范围进行严格检查。
        - 逻辑清晰度: 代码逻辑被重构得更易于理解，并添加了详细的注释来
          解释 `einsum` 字符串的构建规则。
        - 边界条件处理: 正确处理了 `num_total_qubits` 超过 26（字母表大小）
          的理论限制。

        Args:
            num_total_qubits (int): 
                系统的总量子比特数。
            qubits_to_trace (List[int]): 
                要从量子态中迹掉的量子比特索引列表。

        Returns:
            str: 
                一个格式正确的 `einsum` 下标字符串。

        Raises:
            TypeError: 如果 `num_total_qubits` 或 `qubits_to_trace` 的类型不正确。
            ValueError: 如果 `num_total_qubits` 无效，或 `qubits_to_trace` 包含无效索引。
        """
        log_prefix = "QuantumState._generate_einsum_string_for_density_matrix_partial_trace"
        
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(num_total_qubits, int) or not (0 <= num_total_qubits <= 26):
            self._internal_logger.error(f"[{log_prefix}] Invalid 'num_total_qubits' ({num_total_qubits}). Must be an integer between 0 and 26.")
            raise ValueError(
                f"Number of qubits ({num_total_qubits}) for einsum string generation "
                "must be between 0 and 26 for distinct alphabetical indices."
            )
        
        if not isinstance(qubits_to_trace, list) or not all(isinstance(q, int) and 0 <= q < num_total_qubits for q in qubits_to_trace):
            self._internal_logger.error(f"[{log_prefix}] Invalid 'qubits_to_trace' list {qubits_to_trace}. Must be a list of valid qubit indices.")
            raise ValueError("Invalid 'qubits_to_trace' list.")

        # --- 步骤 2: 生成 bra 和 ket 的索引字符 ---
        # 约定：小写字母用于 bra (第一个索引维度)，大写字母用于 ket (第二个索引维度)。
        # bra_indices_chars[i] 对应于第 i 个量子比特的 bra 索引。
        
        # [核心修复]
        # 错误代码: bra_indices_chars = [self.builtins.chr(self.builtins.ord('a') + i) for i in self.builtins.range(num_total_qubits)]
        # 修正后的代码: 直接使用内置函数
        bra_indices_chars = [chr(ord('a') + i) for i in range(num_total_qubits)]
        ket_indices_chars = [chr(ord('A') + i) for i in range(num_total_qubits)]

        # --- 步骤 3: 构建 einsum 输入字符串 ---
        
        # a) 构建 bra 部分的下标字符串 (e.g., 'abc...')
        input_str_bra_part = "".join(bra_indices_chars)
        
        # b) 构建 ket 部分的下标字符串。对于要迹掉的比特，其 ket 下标
        #    必须与 bra 下标相同，以指示 einsum 进行求和。
        # [核心修复]
        ket_indices_chars_modified = list(ket_indices_chars)
        for qubit_index_to_trace in qubits_to_trace:
            # 将要迹掉的比特的 ket 字符替换为对应的 bra 字符
            ket_indices_chars_modified[qubit_index_to_trace] = bra_indices_chars[qubit_index_to_trace]
        input_str_ket_part_modified = "".join(ket_indices_chars_modified)
        
        # c) 组合成完整的输入字符串，例如 'abcABc'
        einsum_input_str = input_str_bra_part + input_str_ket_part_modified
        
        # --- 步骤 4: 构建 einsum 输出字符串 ---
        
        # 输出字符串只包含那些被保留的比特的 bra 和 ket 索引。
        output_bra_chars: List[str] = []
        output_ket_chars: List[str] = []
        
        # [核心修复]
        for i in range(num_total_qubits):
            if i not in qubits_to_trace:
                # 如果比特 i 被保留，则将其 bra 和 ket 字符添加到输出列表中
                output_bra_chars.append(bra_indices_chars[i])
                output_ket_chars.append(ket_indices_chars[i])
        
        # d) 组合成最终的输出字符串，例如 'aA'
        einsum_output_str = "".join(output_bra_chars) + "".join(output_ket_chars)

        # --- 步骤 5: 组合成完整的 einsum 字符串并返回 ---
        final_einsum_string = f"{einsum_input_str}->{einsum_output_str}"
        self._internal_logger.debug(f"[{log_prefix}] Generated einsum string: '{final_einsum_string}'")
        
        return final_einsum_string

    
    
    def get_bloch_vector(self, qubit_index: int) -> Tuple[float, float, float]:
        """
        [健壮性改进版] 计算指定单个量子比特的布洛赫矢量 (rx, ry, rz)。

        Args:
            qubit_index (int): 目标量子比特的索引。

        Returns:
            Tuple[float, float, float]: 布洛赫矢量 (rx, ry, rz)。

        Raises:
            ValueError: 如果 `qubit_index` 无效。
            RuntimeError: 如果底层计算失败。
        """
        log_prefix = f"QuantumState.get_bloch_vector(Qubit={qubit_index})"
        self._internal_logger.debug(f"[{log_prefix}] Starting Bloch vector calculation.")
        
        # --- 1. 输入验证 ---
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"[{log_prefix}] Invalid 'qubit_index' {qubit_index} for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid qubit_index {qubit_index} for a system of {self.num_qubits} qubits.")

        try:
            # --- 2. 执行部分迹，获取单比特约化密度矩阵 ---
            qubits_to_trace_out = [q for q in self.builtins.range(self.num_qubits) if q != qubit_index]
            
            rho_q = self._partial_trace(qubits_to_trace_out)
            
            # [健壮性改进] 检查 rho_q 的形状是否正确 (2x2)
            rho_q_shape = rho_q.shape if hasattr(rho_q, 'shape') else (self.builtins.len(rho_q), self.builtins.len(rho_q[0]))
            if rho_q_shape != (2, 2):
                self._internal_logger.error(f"[{log_prefix}] Partial trace did not result in a 2x2 density matrix. Shape: {rho_q_shape}.")
                raise RuntimeError("Partial trace did not result in a 2x2 density matrix.")

            # --- 3. 计算 Pauli 期望值 (布洛赫矢量分量) ---
            rx = self.get_expectation_value_on_subsystem(rho_q, self._SIGMA_X_LIST)
            ry = self.get_expectation_value_on_subsystem(rho_q, self._SIGMA_Y_LIST)
            rz = self.get_expectation_value_on_subsystem(rho_q, self._SIGMA_Z_LIST)
            
            # [健壮性改进] 裁剪结果到 [-1, 1] 范围
            rx = self._backend.clip(rx, -1.0, 1.0)
            ry = self._backend.clip(ry, -1.0, 1.0)
            rz = self._backend.clip(rz, -1.0, 1.0)

            return (float(rx), float(ry), float(rz))

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] An error occurred during Bloch vector calculation: {e}", exc_info=True)
            raise RuntimeError(f"Bloch vector calculation failed: {e}") from e

    # --- 辅助方法 ---
    def _add_history_log(self, 
                        gate_name: str, 
                        targets: Optional[List[Any]] = None, 
                        params: Optional[Dict[str, Any]] = None,
                        **kwargs):
        """
        [健壮性改进版] [v1.5.8 builtins fix] 将一个操作的详细信息记录到 `gate_application_history` 列表中。
        此版本修复了因错误调用 self.builtins 而导致的 AttributeError，并增强了对不可序列化参数的处理。

        此方法负责为每个应用于 `QuantumState` 的操作创建一个历史记录条目。
        它会合并所有参数，并以一种安全的方式处理可能无法被深拷贝（deepcopy）
        的复杂对象（如函数或模块），确保日志记录功能不会意外地使程序崩溃。

        核心增强功能:
        - 错误修复: 移除了对不存在的 `self.builtins` 的调用，直接使用 Python 
          内置的 `len()` 函数来管理历史记录列表的长度。
        - 严格的输入验证: 确保 `gate_name` 是一个有效的非空字符串。
        - 安全的参数处理: 当 `targets` 或 `parameters` 中包含无法被 `deepcopy`
          的复杂对象（例如，一个函数或一个包含模块引用的对象）时，会捕获
          `TypeError`，并安全地回退到使用该对象的字符串表示 (`str()`) 
          来进行记录。这使得历史记录功能对于包含回调函数或复杂数据结构的
          指令（如 `quantum_subroutine`）同样健壮。
        - 内部状态保护: 在添加新条目之前，会检查 `self.gate_application_history`
          是否存在且为列表类型。如果不是，会记录一个严重警告并将其重置为
          一个空列表，防止因状态损坏导致的进一步错误。
        - 清晰的日志与返回: 提供了详细的内部调试日志，并返回一个布尔值来
          明确表示日志记录是否成功。

        Args:
            gate_name (str): 
                操作的名称。
            targets (Optional[List[Any]], optional): 
                操作的目标量子比特或其他目标。默认为 None。
            params (Optional[Dict[str, Any]], optional): 
                操作的参数字典。默认为 None。
            **kwargs: 
                任何额外的关键字参数，将被合并到 `params` 中。

        Returns:
            bool: 如果日志记录成功，则返回 `True`，否则返回 `False`。
        """
        log_prefix = f"QuantumState._add_history_log(Op: '{gate_name}')"

        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(gate_name, str) or not gate_name.strip():
            self._internal_logger.error(f"[{log_prefix}] Attempted to log history with an invalid operation name: '{gate_name}'. History entry was not recorded.")
            return False # 表示未成功记录日志
        
        history_entry: Dict[str, Any] = {
            "operation_name": gate_name,
            "timestamp_utc_iso": datetime.now(timezone.utc).isoformat()
        }

        # --- 步骤 2: 安全地处理 targets 参数 ---
        if targets is not None:
            if isinstance(targets, list):
                # 尝试深拷贝，但如果内容不可拷贝（如包含函数），则回退到字符串表示
                try:
                    history_entry["targets"] = copy.deepcopy(targets)
                except TypeError as e:
                    self._internal_logger.warning(f"[{log_prefix}] Could not deepcopy 'targets' for history log due to uncopyable content: {e}. Storing a string representation instead.")
                    history_entry["targets"] = str(targets)
            else:
                self._internal_logger.warning(f"[{log_prefix}] 'targets' argument was not a list (got {type(targets).__name__}). Storing a string representation.")
                history_entry["targets"] = str(targets)
        
        # --- 步骤 3: 安全地处理 params 和 kwargs ---
        all_params = (params or {}).copy()
        all_params.update(kwargs)
        if all_params:
            try:
                # 同样，尝试深拷贝，失败则回退
                history_entry["parameters"] = copy.deepcopy(all_params)
            except TypeError as e:
                self._internal_logger.warning(f"[{log_prefix}] Could not deepcopy 'parameters' for history log due to uncopyable content: {e}. Storing a string representation instead.")
                history_entry["parameters"] = str(all_params)

        # --- 步骤 4: 记录日志并管理历史记录列表的长度 ---
        # [健壮性改进] 检查 self.gate_application_history 的存在性和类型
        if not hasattr(self, 'gate_application_history') or not isinstance(self.gate_application_history, list):
            self._internal_logger.critical(
                f"[{log_prefix}] Attribute 'gate_application_history' is missing or not a list. "
                "Resetting to an empty list. This may indicate state corruption."
            )
            self.gate_application_history = []
        
        try:
            self.gate_application_history.append(history_entry)
            
            # [核心修复]
            # 错误代码: if self.builtins.len(self.gate_application_history) > self.MAX_GATE_HISTORY:
            # 修正后的代码: 直接使用内置的 len() 函数
            if len(self.gate_application_history) > self.MAX_GATE_HISTORY:
                # 移除最旧的条目以保持历史记录列表的大小
                self.gate_application_history.pop(0) 
            
            self._update_timestamp() # 更新状态的最后修改时间
            return True # 成功记录日志

        except Exception as e_append:
            self._internal_logger.error(
                f"[{log_prefix}] Unexpected error while appending to gate_application_history: {e_append}",
                exc_info=True
            )
            return False # 记录日志失败
    
    
    def _update_timestamp(self):
        """
        [健壮性改进版] 更新量子态的“最后一次显著状态变化”的时间戳。
        """
        log_prefix = f"QuantumState._update_timestamp(N={self.num_qubits})"
        
        try:
            now_utc = datetime.now(timezone.utc)
            new_timestamp_utc_iso = now_utc.isoformat()
            self.last_significant_update_timestamp_utc_iso = new_timestamp_utc_iso
            self._internal_logger.debug(
                f"[{log_prefix}] 'last_significant_update_timestamp_utc_iso' has been updated to: {new_timestamp_utc_iso}"
            )
            
        except Exception as e_update_ts:
            self._internal_logger.error(
                f"[{log_prefix}] An unexpected error occurred while updating the timestamp: {e_update_ts}",
                exc_info=True
            )

    
    
    def _switch_to_density_matrix_mode(self, reason: str = "Unknown"):
        """
        [超级转换函数 - 最终修正版] 将量子态从 'statevector' 模式不可逆地切换到 'density_matrix' 模式。
        
        此版本通过将哈希函数移入类内部并使用 `self._get_circuit_hash` 来调用，
        彻底解决了 NameError 的作用域问题。
        """
        # --- 步骤 1: 幂等性检查 ---
        if self._simulation_mode == 'density_matrix':
            self._internal_logger.debug("Already in 'density_matrix' mode. No switch needed.")
            return

        log_prefix = f"QuantumState._switch_to_density_matrix_mode(N={self.num_qubits})"
        self._internal_logger.warning(
            f"[{log_prefix}] Initiating expensive and permanent switch from 'statevector' to 'density_matrix' mode. "
            f"Reason: {reason}. All subsequent operations will be non-lazy and operate on a density matrix."
        )

        try:
            # --- 步骤 2: 确保态矢量已计算 ---
            self._expand_to_statevector()
            
            if self._cached_statevector_entity is None or self._cached_statevector_entity._state_vector is None:
                raise RuntimeError("State vector not available for mode switch: expansion failed or resulted in None.")
            
            vec = self._cached_statevector_entity._state_vector
            
            # --- 步骤 3: 记录可回溯的转换日志 ---
            self.mode_transition_log = {
                "timestamp_utc_iso": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "source_state_purity": 1.0,
                # [核心修复] 调用本类的 _get_circuit_hash 方法
                "source_circuit_hash": self._get_circuit_hash(self.circuit)
            }
            
            # --- 步骤 4: 计算密度矩阵 ρ = |ψ⟩⟨ψ| ---
            self._internal_logger.debug(f"[{log_prefix}] Computing outer product to generate density matrix...")
            
            if isinstance(vec, list):
                vec_conj = [v.conjugate() for v in vec]
            else:
                vec_conj = vec.conj()
            
            rho = self._backend.outer(vec, vec_conj)
            
            # --- 步骤 5: 更新状态并清理旧的缓存 ---
            self._density_matrix = rho
            self._simulation_mode = 'density_matrix'
            self._cached_density_matrix_entity = _DensityMatrixEntity(self.num_qubits, self._backend, self._density_matrix)
            self._cached_statevector_entity = None
            self.circuit.instructions = []
            self._is_cache_valid = False 

            self._internal_logger.info(f"[{log_prefix}] Successfully switched to 'density_matrix' mode.")
            
            # --- 步骤 6: 归一化 ---
            self.normalize()

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] Failed to switch to density_matrix mode: {e}", exc_info=True)
            
            # [失败安全]
            self._simulation_mode = 'density_matrix'
            self._density_matrix = None
            self._cached_statevector_entity = None
            self._cached_density_matrix_entity = None
            self._is_cache_valid = False
            self.mode_transition_log = {
                "timestamp_utc_iso": datetime.now(timezone.utc).isoformat(),
                "reason": "Failed transition",
                "error": str(e)
            }
            
            raise RuntimeError(f"Failed to transition to density_matrix mode: {e}") from e
    def run_circuit(self, circuit: 'QuantumCircuit', 
                    noise_model: Optional['NoiseModel'] = None,
                    topology: Optional[Dict[int, List[int]]] = None):
        """
        [健壮性改进版 - 模式切换优化] 在一个惰性/双模式量子态上“执行”一个量子线路中的所有指令。

        此方法是 QuantumState 的核心执行引擎。它会遍历传入电路中的每一条指令，并根据
        当前的模拟模式 (`statevector` 或 `density_matrix`) 和噪声模型的存在与否，
        智能地决定如何应用这些指令。

        核心工作流程:
        1.  **输入验证**: 严格检查所有输入参数 (`circuit`, `noise_model`, `topology`) 的类型和有效性。
        2.  **模式切换**: 如果提供了 `noise_model` 且当前处于 `'statevector'` 模式，
            会立即调用 `_switch_to_density_matrix_mode` 强制切换到密度矩阵模式。
        3.  **指令循环**: 遍历电路中的每一条指令。
            a.  **经典条件检查**: 如果指令包含经典条件，会检查 `_classical_registers`
                并决定是否跳过该指令。
            b.  **噪声查询**: 如果处于密度矩阵模式且提供了噪声模型，会调用
                `noise_model.get_noise_for_op` 来获取该指令对应的噪声。
            c.  **指令执行**:
                -   在 `'density_matrix'` 模式下，会将理想操作（可能已被相干误差修改）
                    和非相干噪声通道依次委托给 `_cached_density_matrix_entity` 执行。
                -   在 `'statevector'` 模式下，会区分“酉操作”和“断点操作”：
                    -   对于酉操作，如果缓存有效，则立即在 `_StateVectorEntity` 上
                        执行（即时演化）；否则，将其追加到 `self.circuit` 中（惰性求值）。
                    -   对于测量等“断点操作”，会先将之前累积的酉操作块应用到状态上
                        （触发展开），然后再执行断点操作。

        Args:
            circuit (QuantumCircuit): 要执行的量子线路。
            noise_model (Optional['NoiseModel']): 可选的噪声模型实例。如果提供，模拟将包含噪声效应。
            topology (Optional[Dict[int, List[int]]]): 可选的硬件拓扑图。如果提供，它将被传递给内部的
                                                     `nexus_optimizer` 以执行拓扑感知的编译。

        Raises:
            TypeError: 如果输入参数类型不正确。
            ValueError: 如果线路比特数不匹配或指令无效。
            RuntimeError: 如果线路执行失败。
        """
        log_prefix = f"QuantumState.run_circuit(N={self.num_qubits})"
        
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(circuit, QuantumCircuit):
            self._internal_logger.error(f"[{log_prefix}] Input 'circuit' must be a QuantumCircuit instance, but got {type(circuit).__name__}.")
            raise TypeError(f"Input 'circuit' must be a QuantumCircuit instance.")
        if circuit.num_qubits > self.num_qubits:
            self._internal_logger.error(f"[{log_prefix}] State ({self.num_qubits} qubits) cannot execute a circuit for more qubits ({circuit.num_qubits} qubits).")
            raise ValueError(
                f"State ({self.num_qubits} qubits) cannot execute a circuit designed for more qubits "
                f"({circuit.num_qubits} qubits). Max allowed is {self.num_qubits}."
            )
        if noise_model is not None and not isinstance(noise_model, NoiseModel):
             self._internal_logger.error(f"[{log_prefix}] 'noise_model' must be a NoiseModel instance or None, got {type(noise_model).__name__}.")
             raise TypeError(f"'noise_model' must be a NoiseModel instance or None.")
        if topology is not None and not isinstance(topology, dict):
             self._internal_logger.error(f"[{log_prefix}] 'topology' must be a dictionary or None, got {type(topology).__name__}.")
             raise TypeError(f"'topology' must be a dictionary or None.")

        noise_info_str = f"with noise model '{type(noise_model).__name__}'" if noise_model else "ideally (no noise)"
        topo_info_str = "with topology optimization" if topology else "without topology optimization"
        self._internal_logger.info(
            f"[{log_prefix}] Running circuit '{circuit.description or 'unnamed'}' with "
            f"{len(circuit.instructions)} instructions {noise_info_str} and {topo_info_str}. "
            f"Current mode: '{self._simulation_mode}'."
        )

        # --- 步骤 2: 模式切换 (如果需要) ---
        if noise_model is not None and self._simulation_mode == 'statevector':
            self._switch_to_density_matrix_mode(reason=f"Execution with noise model '{type(noise_model).__name__}'")
        
        # --- 步骤 3: 准备指令循环 ---
        # 用于在 statevector 模式下累积连续的酉操作
        unitary_block_instructions: List[Tuple[Any, ...]] = []

        instr_index, instruction = -1, None # 初始化以确保在异常情况下可用
        try:
            # --- 步骤 4: 遍历并执行电路中的每一条指令 ---
            for instr_index, instruction in enumerate(circuit.instructions):
                # a) 解析指令
                gate_name = instruction[0]
                op_kwargs = instruction[-1].copy() if isinstance(instruction[-1], dict) else {}
                op_args = list(instruction[1:-1]) if isinstance(instruction[-1], dict) else list(instruction[1:])
                
                # b) 经典条件检查
                condition = op_kwargs.pop('condition', None)
                if condition is not None:
                    cr_index, expected_value = condition
                    if self._classical_registers.get(cr_index) != expected_value:
                        self._internal_logger.debug(f"[{log_prefix}] Skipping conditional gate '{gate_name}' at index {instr_index} due to condition mismatch.")
                        continue
                
                # c) 噪声模型集成 (仅在密度矩阵模式下)
                noise_info: Dict[str, Any] = {}
                if self._simulation_mode == 'density_matrix' and noise_model:
                    # 获取门操作涉及的所有量子比特，用于噪声查询
                    qubits_involved = [q for arg in op_args for q in (arg if isinstance(arg, list) else [arg]) if isinstance(q, int)]
                    if qubits_involved:
                        noise_info = noise_model.get_noise_for_op(gate_name, qubits_involved)
                
                # d) 执行指令
                if self._simulation_mode == 'density_matrix':
                    # 路径 A: 密度矩阵模式下的执行
                    replacement_circuit = noise_info.get('coherent_unitary_replacement', {}).get('circuit')
                    if replacement_circuit:
                        # 如果噪声模型要求完全替换，则递归调用 run_circuit
                        self.run_circuit(replacement_circuit, noise_model=None, topology=topology) 
                    else:
                        # 否则，应用相干误差（如果存在）并执行理想门
                        coherent_error_info = noise_info.get('coherent_error')
                        if coherent_error_info and 'angle_error' in coherent_error_info:
                            try:
                                # 寻找最后一个数值参数并修改
                                target_idx_to_modify = -1
                                for idx in range(len(op_args) - 1, -1, -1):
                                    if isinstance(op_args[idx], (float, int)):
                                        target_idx_to_modify = idx
                                        break
                                if target_idx_to_modify != -1:
                                    op_args[target_idx_to_modify] += coherent_error_info['angle_error']
                            except Exception as e:
                                self._internal_logger.warning(f"[{log_prefix}] Error applying coherent angle error: {e}")
                        
                        self._execute_gate_logic(gate_name, op_args, op_kwargs, topology=topology)
                    
                    # 应用非相干后置噪声
                    if 'incoherent_post_op' in noise_info:
                        for channel_info in noise_info['incoherent_post_op']:
                            self.apply_quantum_channel(topology=topology, **channel_info)

                elif self._simulation_mode == 'statevector':
                    # 路径 B: 态矢量模式下的惰性/即时执行
                    is_breakpoint = (gate_name in ('simulate_measurement', 'apply_quantum_channel', 'quantum_subroutine'))

                    if is_breakpoint:
                        # 遇到断点，先处理累积的酉操作块
                        if unitary_block_instructions:
                            self.circuit.instructions.extend(unitary_block_instructions)
                            unitary_block_instructions = [] 
                            self._invalidate_cache()
                        
                        # 然后执行断点操作
                        if gate_name == "simulate_measurement":
                            self.simulate_measurement(op_args[0] if op_args else None, topology=topology, **op_kwargs)
                        elif gate_name == "apply_quantum_channel":
                            self.apply_quantum_channel(op_args[0], op_args[1], op_kwargs.get('params', {}), topology=topology, **op_kwargs)
                        elif gate_name == "quantum_subroutine":
                            self.quantum_subroutine(op_args[0], op_args[1], topology=topology, **op_kwargs)
                    
                    else: # 如果是酉门
                        # 在 statevector 模式下，_execute_gate_logic 内部会处理惰性/即时逻辑
                        self._execute_gate_logic(gate_name, op_args, op_kwargs, topology=topology)

                else:
                    raise ValueError(f"Unknown simulation mode: '{self._simulation_mode}'.")

        except Exception as e:
            # --- 步骤 5: 健壮的全局异常捕获 ---
            if instruction is None: instruction = "N/A (error before loop start)"
            if instr_index is None: instr_index = -1
            self._internal_logger.critical(f"[{log_prefix}] Failed to execute instruction #{instr_index} ('{instruction}'): {e}", exc_info=True)
            raise RuntimeError(f"Circuit execution failed at instruction '{instruction}' (index {instr_index}).") from e

        # --- 步骤 6: 线路执行结束后的收尾工作 ---
        # 如果在 statevector 模式下，有剩余的酉操作块未处理
        if self._simulation_mode == 'statevector' and unitary_block_instructions:
            self.circuit.instructions.extend(unitary_block_instructions)
            self._invalidate_cache()

        self._internal_logger.info(f"[{log_prefix}] All instructions processed successfully. Final mode: '{self._simulation_mode}'.")
    
    
    # --- VQA, 动力学, 分析 API ---
    def get_marginal_probabilities(self, qubit_index_to_keep: int) -> Optional[Tuple[float, float]]:
        """
        [健壮性改进版] 计算指定单个量子比特的边际概率分布 P(0) 和 P(1)。

        Args:
            qubit_index_to_keep (int): 要保留（计算边际概率）的量子比特索引。

        Returns:
            Optional[Tuple[float, float]]: 
                一个元组 `(prob_0, prob_1)`。如果计算失败，返回 `None`。

        Raises:
            ValueError: 如果 `qubit_index_to_keep` 无效。
            RuntimeError: 如果底层计算失败。
        """
        log_prefix = f"QuantumState.MarginalProbs(N={self.num_qubits}, KeepQ={qubit_index_to_keep})"
        
        # --- 1. 输入验证 ---
        if not isinstance(qubit_index_to_keep, int) or not (0 <= qubit_index_to_keep < self.num_qubits):
            self._internal_logger.error(f"[{log_prefix}] Invalid 'qubit_index_to_keep' {qubit_index_to_keep}.")
            raise ValueError(f"Invalid target qubit index: {qubit_index_to_keep}")
        
        try:
            # --- 2. 执行部分迹，获取单比特约化密度矩阵 ---
            qubits_to_trace_out = [i for i in self._backend.builtins.range(self.num_qubits) if i != qubit_index_to_keep]
            rho_q = self._partial_trace(qubits_to_trace_out)
            
            # [健壮性改进] 检查 rho_q 的形状是否正确 (2x2)
            rho_q_shape = rho_q.shape if hasattr(rho_q, 'shape') else (self._backend.builtins.len(rho_q), self._backend.builtins.len(rho_q[0]))
            if rho_q_shape != (2, 2):
                self._internal_logger.error(f"[{log_prefix}] Partial trace did not result in a 2x2 density matrix. Shape: {rho_q_shape}.")
                raise RuntimeError("Partial trace did not result in a 2x2 density matrix.")

            # --- 3. 提取对角线元素作为概率 ---
            prob_0_raw = rho_q[0][0].real if isinstance(rho_q, list) else rho_q[0, 0].real
            prob_1_raw = rho_q[1][1].real if isinstance(rho_q, list) else rho_q[1, 1].real

            # [健壮性改进] 裁剪概率值到 [0, 1] 范围，并归一化
            prob_0 = float(self._backend.clip(prob_0_raw, 0.0, 1.0))
            prob_1 = float(self._backend.clip(prob_1_raw, 0.0, 1.0))

            prob_sum = prob_0 + prob_1
            if not self._backend.isclose(prob_sum, 1.0, atol=1e-7):
                self._internal_logger.warning(
                    f"[{log_prefix}] Sum of marginal probabilities ({prob_sum:.12f}) is not 1.0. Renormalizing."
                )
                if prob_sum > 1e-12: # 避免除以零
                    prob_0 /= prob_sum
                    prob_1 /= prob_sum
                else:
                    # 如果总概率接近0，则无法可靠归一化，返回默认值
                    self._internal_logger.warning(f"[{log_prefix}] Sum of marginal probabilities is near zero. Returning (0.5, 0.5) as fallback.")
                    return (0.5, 0.5) # 返回一个合理的默认值

            return (prob_0, prob_1)

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] An unknown error occurred while calculating marginal probabilities: {e}", exc_info=True)
            raise RuntimeError(f"Marginal probability calculation failed: {e}") from e
    def calculate_von_neumann_entropy(self, qubits_to_partition: List[int]) -> float:
        """
        [健壮性改进版] 计算指定子系统的冯·诺依曼纠缠熵 S(ρ_A) = -Tr(ρ_A log₂(ρ_A))。

        此方法量化了一个量子子系统 `A` 与系统其余部分的纠缠程度。
        其计算步骤如下：
        1.  通过对系统的其余部分执行部分迹，获得子系统 `A` 的约化密度矩阵 `ρ_A`。
        2.  计算 `ρ_A` 的所有特征值 `{λ_i}`。
        3.  使用香农熵的公式计算冯·诺依曼熵：`S = -Σ_i (λ_i * log₂(λ_i))`。

        核心增强功能:
        - 错误修复: 移除了对不存在的 `self.builtins` 的调用，直接使用 Python 
          内置的 `len()` 函数，解决了 `AttributeError`。
        - 严格的输入验证: 在执行任何计算之前，对输入的 `qubits_to_partition`
          列表进行全面的验证，确保其非空、元素为有效 qubit 索引且不重复。
        - 健壮的中间结果检查: 在部分迹操作后，验证生成的约化密度矩阵是否为
          预期的方阵，防止后续计算因形状错误而失败。
        - 数值稳定性: 
          - 在计算熵时，会忽略接近于零的特征值，以避免 `log2(0)` 导致的数学错误。
          - 明确地将特征值裁剪到 `[0.0, 1.0]` 区间，以处理由于浮点数精度误差
            可能导致的微小负值或大于1的值。
        - 后端无关性: 所有数值计算（部分迹、特征值分解、对数）都通过 `self._backend` 
          的抽象接口进行，确保此方法能在 PurePython 和 CuPy 后端下一致地工作。
        - 清晰的错误处理: 提供了详细的日志信息，并在发生任何内部错误时，捕获
          异常并重新包装为带有清晰上下文的 `RuntimeError`。

        Args:
            qubits_to_partition (List[int]): 
                一个整数列表，定义了要计算熵的子系统 `A` 所包含的量子比特。

        Returns:
            float: 
                计算出的冯·诺依曼熵（一个非负实数）。对于纯态，熵为0；对于最大纠缠态，
                熵为 `log₂(dim(A))`。

        Raises:
            TypeError: 如果 `qubits_to_partition` 不是列表或包含非整数元素。
            ValueError: 如果 `qubits_to_partition` 为空、包含无效或重复的索引。
            RuntimeError: 如果底层计算（如部分迹或特征值分解）失败。
        """
        log_prefix = f"QuantumState.calculate_von_neumann_entropy(Partition={qubits_to_partition})"
        self._internal_logger.debug(f"[{log_prefix}] Starting entropy calculation.")

        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(qubits_to_partition, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits_to_partition):
            self._internal_logger.error(f"[{log_prefix}] 'qubits_to_partition' must be a non-empty list of valid qubit indices, got {qubits_to_partition}.")
            raise ValueError("qubits_to_partition must be a non-empty list of valid qubit indices.")
        if not qubits_to_partition:
            self._internal_logger.error(f"[{log_prefix}] 'qubits_to_partition' cannot be empty.")
            raise ValueError("'qubits_to_partition' cannot be empty.")
        
        if len(set(qubits_to_partition)) != len(qubits_to_partition):
            self._internal_logger.error(f"[{log_prefix}] 'qubits_to_partition' contains duplicate qubit indices: {qubits_to_partition}.")
            raise ValueError("qubits_to_partition contains duplicate qubit indices.")

        try:
            # --- 步骤 2: 执行部分迹，获取约化密度矩阵 ρ_A ---
            qubits_to_trace_out = [q for q in range(self.num_qubits) if q not in qubits_to_partition]
            
            rho_A = self._partial_trace(qubits_to_trace_out)
            
            # [健壮性改进] 检查 rho_A 是否为方阵
            # [核心修复]
            # 错误代码: if (hasattr(rho_A, 'shape') and rho_A.shape[0] != rho_A.shape[1]) or (isinstance(rho_A, list) and rho_A and self._backend.builtins.len(rho_A) != self._backend.builtins.len(rho_A[0])): 
            # 修正后的代码: 直接使用内置的 len()
            if (hasattr(rho_A, 'shape') and rho_A.shape[0] != rho_A.shape[1]) or \
                (isinstance(rho_A, list) and rho_A and len(rho_A) != len(rho_A[0])): 
                shape_str = rho_A.shape if hasattr(rho_A, 'shape') else (len(rho_A), len(rho_A[0]))
                self._internal_logger.error(f"[{log_prefix}] Partial trace did not result in a square matrix. Shape: {shape_str}.")
                raise RuntimeError("Internal error: Partial trace did not produce a square matrix.")

            # --- 步骤 3: 计算 ρ_A 的特征值 ---
            # eigvalsh 专门用于厄米矩阵，返回实数特征值，且效率更高
            eigenvalues = self._backend.eigvalsh(rho_A)
            
            # --- 步骤 4: 计算冯·诺依曼熵 S = -Σ_i (λ_i * log₂(λ_i)) ---
            entropy = 0.0
            
            # 无论是 PurePython 列表还是 CuPy 数组，都可以直接迭代
            for eigenvalue in eigenvalues:
                # 确保 eigenvalue 是标准的 Python float
                eigenvalue_float = float(eigenvalue)
                
                # [健壮性改进] 裁剪特征值到正数区间，以处理浮点误差可能导致的微小负值
                eigenvalue_clipped = self._backend.clip(eigenvalue_float, 0.0, 1.0)
                
                # 只有当特征值显著大于零时才进行计算，以避免 log2(0)
                if eigenvalue_clipped > 1e-12: 
                    entropy -= eigenvalue_clipped * self._backend.log2(eigenvalue_clipped)
            
            # --- 步骤 5: 返回最终结果 ---
            return float(entropy)

        except Exception as e:
            # 捕获所有可能的底层计算错误
            self._internal_logger.critical(f"[{log_prefix}] An error occurred during entropy calculation: {e}", exc_info=True)
            raise RuntimeError(f"Entanglement entropy calculation failed: {e}") from e

    def get_bloch_vector(self, qubit_index: int) -> Tuple[float, float, float]:
        """
        [健壮性改进版] 计算指定单个量子比特的布洛赫矢量 (rx, ry, rz)。

        此方法通过执行部分迹来获得目标量子比特的约化密度矩阵，
        然后计算 Pauli 算子 X, Y, Z 在该约化密度矩阵上的期望值，
        这三个期望值即构成了布洛赫矢量的三个分量。

        核心增强功能:
        - 错误修复: 修正了因错误调用 `self.builtins` 而导致的 `AttributeError`。
          现在通过 `self._backend.builtins` 正确访问后端抽象的内置函数。
        - 严格的输入验证: 确保 `qubit_index` 是一个在系统范围内的有效整数。
        - 健壮的中间结果检查: 在部分迹操作后，验证生成的约化密度矩阵是否为
          预期的 2x2 矩阵，防止后续计算因形状错误而失败。
        - 数值稳定性: 在计算最终分量后，使用后端的 `clip` 方法将结果裁剪到
          物理解释所允许的 `[-1.0, 1.0]` 区间内，以处理由于浮点数精度误差
          可能导致的微小越界。
        - 清晰的错误处理与日志记录: 提供了详细的日志信息，并在发生任何
          内部错误时，捕获异常并重新包装为带有清晰上下文的 `RuntimeError`。
        - 类型保证: 确保最终返回的元组中的每个元素都是标准的 Python `float` 类型。

        Args:
            qubit_index (int): 目标量子比特的索引。

        Returns:
            Tuple[float, float, float]: 布洛赫矢量 (rx, ry, rz)。

        Raises:
            ValueError: 如果 `qubit_index` 无效。
            RuntimeError: 如果底层计算（如部分迹或期望值计算）失败。
        """
        log_prefix = f"QuantumState.get_bloch_vector(Qubit={qubit_index})"
        self._internal_logger.debug(f"[{log_prefix}] Starting Bloch vector calculation.")
        
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            self._internal_logger.error(f"[{log_prefix}] Invalid 'qubit_index' {qubit_index} for a system of {self.num_qubits} qubits.")
            raise ValueError(f"Invalid qubit_index {qubit_index} for a system of {self.num_qubits} qubits.")

        try:
            # --- 步骤 2: 执行部分迹，获取单比特约化密度矩阵 ---
            
            # [核心修复] 
            # 错误代码: qubits_to_trace_out = [q for q in self.builtins.range(self.num_qubits) if q != qubit_index]
            # 修正后的代码: 通过 self._backend 访问 builtins 属性
            qubits_to_trace_out = [q for q in self._backend.builtins.range(self.num_qubits) if q != qubit_index]
            
            rho_q = self._partial_trace(qubits_to_trace_out)
            
            # [健壮性改进] 检查 rho_q 的形状是否为预期的 2x2
            rho_q_shape = rho_q.shape if hasattr(rho_q, 'shape') else (self._backend.builtins.len(rho_q), self._backend.builtins.len(rho_q[0]))
            if rho_q_shape != (2, 2):
                self._internal_logger.error(f"[{log_prefix}] Partial trace did not result in a 2x2 density matrix. Actual shape: {rho_q_shape}.")
                raise RuntimeError("Internal error: Partial trace did not produce a 2x2 density matrix.")

            # --- 步骤 3: 计算 Pauli 期望值 (布洛赫矢量分量) ---
            # get_expectation_value_on_subsystem 会处理不同后端的数据类型
            rx = self.get_expectation_value_on_subsystem(rho_q, self._SIGMA_X_LIST)
            ry = self.get_expectation_value_on_subsystem(rho_q, self._SIGMA_Y_LIST)
            rz = self.get_expectation_value_on_subsystem(rho_q, self._SIGMA_Z_LIST)
            
            # --- 步骤 4: [健壮性改进] 裁剪结果到物理允许的 [-1, 1] 范围 ---
            # 这样做可以处理由于浮点数精度误差导致的微小越界 (例如，1.0000000001)
            rx = self._backend.clip(rx, -1.0, 1.0)
            ry = self._backend.clip(ry, -1.0, 1.0)
            rz = self._backend.clip(rz, -1.0, 1.0)

            # --- 步骤 5: 确保返回的是标准的 Python float 元组 ---
            return (float(rx), float(ry), float(rz))

        except Exception as e:
            # 捕获所有可能的底层异常，并包装成一个带有清晰上下文的 RuntimeError
            self._internal_logger.critical(f"[{log_prefix}] An error occurred during Bloch vector calculation: {e}", exc_info=True)
            raise RuntimeError(f"Bloch vector calculation failed: {e}") from e


# ========================================================================
# --- 7. [扩展] 预置的噪声模型 ---
# ========================================================================

class PrebuiltNoiseModels:
    """
    一个命名空间类，用于组织所有预置的、具体的噪声模型。

    这个类本身不应该被实例化。它的作用是作为一个容器，将所有具体的
    `NoiseModel` 子类组织在一起，使得用户可以通过一个清晰的路径
    (e.g., `PrebuiltNoiseModels.HardwareBackend`) 来访问它们。

    这种设计模式在保持单文件结构的同时，实现了逻辑上的模块化。
    """

    class CoherentGateError(NoiseModel):
        """
        一个简单的相干噪声模型，为旋转门引入高斯分布的角度误差。

        这个模型可以用来模拟由于控制脉冲不完美导致的系统性过旋转或欠旋转。
        例如，一个理想的 `RX(π/2)` 门在有此噪声的情况下，可能会被执行为
        `RX(π/2 + ε)`，其中 ε 是一个从高斯分布中随机抽取的小误差。

        由于误差是加在酉算子的参数上的，它保持了演化的幺正性（纯态仍是纯态），
        但会将量子态旋转到错误的位置，因此被称为“相干”误差。
        """
        def __init__(self, angle_error_mean: float = 0.0, angle_error_std: float = 0.01):
            """
            初始化相干门误差模型。

            Args:
                angle_error_mean (float):
                    角度误差的高斯分布均值 (单位：弧度)。一个非零的均值
                    可以模拟系统性的校准偏差。默认为 0.0。
                angle_error_std (float):
                    角度误差的高斯分布标准差 (单位：弧度)。一个非零的标准差
                    可以模拟随机的脉冲波动。默认为 0.01。
            
            Raises:
                ValueError: 如果 `angle_error_std` 为负数。
                TypeError: 如果 `angle_error_mean` 或 `angle_error_std` 不是数值类型。
            """
            super().__init__() # 调用基类初始化，设置logger
            
            # --- 输入验证 ---
            if not isinstance(angle_error_mean, (float, int)):
                raise TypeError(f"CoherentGateError: 'angle_error_mean' must be a numeric type, but got {type(angle_error_mean).__name__}.")
            if not isinstance(angle_error_std, (float, int)):
                raise TypeError(f"CoherentGateError: 'angle_error_std' must be a numeric type, but got {type(angle_error_std).__name__}.")
            if angle_error_std < 0:
                raise ValueError("CoherentGateError: 'angle_error_std' must be a non-negative number.")
            
            self.mean = float(angle_error_mean)
            self.std = float(angle_error_std)
            
            # 定义一组受此噪声模型影响的门操作
            # 这些都是直接有角度参数的旋转门
            self.ROTATION_GATES: set[str] = {
                'rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz', 'p_gate',
                'cp', 'crx', 'cry', 'crz', 'u3_gate' # U3也有角度参数
            }
            self._internal_logger.info(f"CoherentGateError initialized: mean={self.mean:.4e} rad, std={self.std:.4e} rad.")

        def get_noise_for_op(self, op_name: str, qubits: List[int]) -> Dict[str, Any]:
            """
            如果操作是旋转门，则为其生成一个随机的角度误差。

            此方法实现了 `NoiseModel` 的接口。当 `run_circuit` 遇到一个
            旋转门时，它会调用此方法，并获得一个包含 `coherent_error`
            信息的字典，然后用它来修改门的旋转角度参数。

            Args:
                op_name (str): 门操作的名称。
                qubits (List[int]): 门作用的量子比特列表 (此模型中通常未使用，除非门类型与比特数强相关)。

            Returns:
                Dict[str, Any]:
                    一个描述噪声的字典，如果适用，则包含 `coherent_error` 键。
                    例如: `{'coherent_error': {'angle_error': 0.0123}}`。
                    如果 `op_name` 不是旋转门，则返回空字典。
            """
            noise: Dict[str, Any] = {}
            if op_name in self.ROTATION_GATES:
                # 只有当标准差大于零时才引入随机性，否则误差是确定的（等于均值）
                if self.std > 1e-12: # 使用一个小的阈值来避免浮点数比较问题
                    error = random.gauss(self.mean, self.std)
                else:
                    error = self.mean
                
                # 仅当误差的绝对值大于一个很小的阈值时才添加，以避免不必要的操作
                if abs(error) > 1e-12:
                    noise['coherent_error'] = {'angle_error': error}
                    self._internal_logger.debug(f"Applied coherent error {error:.4e} to '{op_name}' on {qubits}.")
                else:
                    self._internal_logger.debug(f"Coherent error for '{op_name}' on {qubits} is negligible, skipping.")
            else:
                self._internal_logger.debug(f"No coherent error defined for non-rotation gate '{op_name}'.")
            
            return noise

    class HardwareBackend(NoiseModel):
        """
        一个模拟真实硬件噪声特性的模型。

        这个模型整合了多种物理上重要的非相干噪声源，旨在提供一个更接近
        现实世界量子计算机（NISQ时代设备）行为的模拟环境。

        它通过一个结构化的字典 `calibration_data` 来配置，该字典模仿了
        真实量子硬件供应商提供的校准数据。

        支持的噪声类型:
        1. T1 能量弛豫: 模拟量子比特自发地从 |1⟩ 态衰变到 |0⟩ 态。
        2. T2 退相位: 模拟量子比特叠加态相位信息的丢失。
        3. 门保真度错误: 模拟门操作本身的不完美性，通常建模为去极化噪声。
        4. 读出错误: 模拟在测量结束时，经典地将 0 读成 1 或将 1 读成 0 的错误。
        """
        def __init__(self, calibration_data: Dict[str, Any]):
            """
            初始化硬件后端噪声模型。

            Args:
                calibration_data (Dict[str, Any]):
                    一个包含硬件参数的字典。
                    示例结构:
                    {
                        'qubits': {
                            0: {'T1': 50e-6, 'T2': 70e-6, 'readout_error': 0.01},
                            1: {'T1': 60e-6, 'T2': 80e-6, 'readout_error': 0.015},
                        },
                        'gates': {
                            'h': {'duration': 50e-9, 'error_rate': 0.001},
                            'cnot': {
                                'duration': 300e-9,
                                'error_rate': 0.05,
                                'qubits': [0, 1]  # 指定此参数只适用于 (0,1) CNOT
                            },
                        }
                    }
            Raises:
                TypeError: 如果 `calibration_data` 不是字典。
                ValueError: 如果 `calibration_data` 的结构或内容无效。
            """
            super().__init__() # 调用基类初始化，设置logger
            
            # --- 输入验证 ---
            if not isinstance(calibration_data, dict):
                raise TypeError("HardwareBackend: 'calibration_data' must be a dictionary.")
            
            # 执行深度验证和预处理校准数据
            self.calib = self._validate_and_preprocess_calibration(calibration_data)
            self._internal_logger.info("HardwareBackend initialized with validated calibration data.")

        def _validate_and_preprocess_calibration(self, raw_calib: Dict[str, Any]) -> Dict[str, Any]:
            """
            对原始校准数据进行深度验证和标准化处理。
            """
            validated_calib: Dict[str, Any] = {'qubits': {}, 'gates': {}}

            # 验证 'qubits' 部分
            raw_qubits = raw_calib.get('qubits', {})
            if not isinstance(raw_qubits, dict):
                raise ValueError("HardwareBackend: 'calibration_data.qubits' must be a dictionary.")
            for q_idx_str, props in raw_qubits.items():
                try:
                    q_idx = int(q_idx_str) # 允许使用字符串键，但内部转为整数
                except ValueError:
                    raise ValueError(f"HardwareBackend: Qubit index '{q_idx_str}' must be an integer.")
                if not isinstance(props, dict):
                    raise ValueError(f"HardwareBackend: Properties for qubit {q_idx} must be a dictionary.")
                
                # 验证T1, T2, readout_error
                validated_props: Dict[str, Union[float, int]] = {}
                for key in ['T1', 'T2', 'readout_error']:
                    val = props.get(key)
                    if val is not None:
                        if not isinstance(val, (float, int)) or val < 0:
                            raise ValueError(f"HardwareBackend: Qubit {q_idx}'s '{key}' must be a non-negative number.")
                        validated_props[key] = float(val)
                validated_calib['qubits'][q_idx] = validated_props

            # 验证 'gates' 部分
            raw_gates = raw_calib.get('gates', {})
            if not isinstance(raw_gates, dict):
                raise ValueError("HardwareBackend: 'calibration_data.gates' must be a dictionary.")
            for gate_name, props in raw_gates.items():
                if not isinstance(gate_name, str) or not gate_name.strip():
                    raise ValueError("HardwareBackend: Gate name must be a non-empty string.")
                if not isinstance(props, dict):
                    raise ValueError(f"HardwareBackend: Properties for gate '{gate_name}' must be a dictionary.")
                
                validated_gate_props: Dict[str, Any] = {}
                for key in ['duration', 'error_rate']:
                    val = props.get(key)
                    if val is not None:
                        if not isinstance(val, (float, int)) or val < 0:
                            raise ValueError(f"HardwareBackend: Gate '{gate_name}'s '{key}' must be a non-negative number.")
                        validated_gate_props[key] = float(val)
                
                # 验证qubits参数
                if 'qubits' in props:
                    q_list = props['qubits']
                    if not isinstance(q_list, list) or not all(isinstance(q, int) for q in q_list):
                        raise ValueError(f"HardwareBackend: Gate '{gate_name}'s 'qubits' must be a list of integers.")
                    validated_gate_props['qubits'] = sorted(list(set(q_list))) # 存储为排序后的唯一列表，方便匹配

                validated_calib['gates'][gate_name] = validated_gate_props

            return validated_calib

        def get_noise_for_op(self, op_name: str, qubits: List[int]) -> Dict[str, Any]:
            """
            根据校准数据，为给定的门操作生成 T1/T2 弛豫噪声和门错误噪声。

            Args:
                op_name (str): 门操作的名称。
                qubits (List[int]): 门作用的量子比特列表。

            Returns:
                Dict[str, Any]:
                    一个描述噪声的字典，包含 'incoherent_post_op' 键。
                    如果门或比特没有校准数据，则不添加相应噪声。
            """
            noise: Dict[str, Any] = {'incoherent_post_op': []}
            
            # --- 步骤 1: 查找与当前操作匹配的门信息 ---
            gate_info: Optional[Dict[str, Any]] = None
            gates_calib = self.calib.get('gates', {})
            
            # a) 优先查找精确匹配的门 (e.g., 'h')
            if op_name in gates_calib and 'qubits' not in gates_calib[op_name]:
                gate_info = gates_calib[op_name]
                self._internal_logger.debug(f"HardwareBackend: Found exact gate calibration for '{op_name}'.")
            
            # b) 如果找不到，尝试匹配特定比特对的门 (e.g., a cnot on [0,1])
            # 确保 qubits 列表已排序，以便与存储的校准数据匹配
            sorted_qubits_tuple = tuple(sorted(qubits))
            if not gate_info:
                for key, info in gates_calib.items():
                    if 'qubits' in info and tuple(info['qubits']) == sorted_qubits_tuple:
                        # 检查门名称是否匹配或者为通配符
                        if key == op_name:
                            gate_info = info
                            self._internal_logger.debug(f"HardwareBackend: Found gate calibration for '{op_name}' on specific qubits {qubits}.")
                            break
                        # 如果需要支持 'any' 门通配符，可以在这里添加逻辑
                        # elif key == 'any_two_qubit_gate' and len(qubits) == 2:
                        #     gate_info = info
                        #     break
            
            # 如果仍然找不到，则假设此门是理想的、无噪声的
            if not gate_info:
                self._internal_logger.debug(f"HardwareBackend: No specific calibration found for gate '{op_name}' on {qubits}. Assuming ideal gate.")
                return {}

            duration: float = gate_info.get('duration', 0.0)
            
            # --- 步骤 2: 模拟 T1/T2 弛豫噪声 ---
            for q in qubits:
                q_props = self.calib.get('qubits', {}).get(q)
                if not q_props:
                    self._internal_logger.debug(f"HardwareBackend: No qubit calibration found for qubit {q}. Skipping T1/T2.")
                    continue
                if duration <= 1e-12: # 如果持续时间太短，弛豫效应可忽略
                    self._internal_logger.debug(f"HardwareBackend: Gate duration ({duration:.2e}s) too short for T1/T2 on qubit {q}.")
                    continue

                # T1 (Amplitude Damping)
                if 'T1' in q_props and q_props['T1'] > 1e-12:
                    prob_t1 = 1.0 - math.exp(-duration / q_props['T1'])
                    if prob_t1 > 1e-12:
                        noise['incoherent_post_op'].append({
                            'channel_type': 'amplitude_damping', 'target_qubits': [q], 'params': {'gamma': prob_t1}
                        })
                        self._internal_logger.debug(f"HardwareBackend: Added T1 (gamma={prob_t1:.4e}) for qubit {q}.")

                # T2 (Phase Damping)
                # T2' = 1 / (1/T2_total - 1/(2*T1))
                # prob_t2 = 1 - exp(-duration / T2_prime)
                # For simplicity, we can use T2_total here if T2 > T1/2
                if 'T2' in q_props and q_props['T2'] > 1e-12:
                    # T2_prime calculation for pure dephasing component
                    if 'T1' in q_props and q_props['T1'] > 1e-12:
                        T2_total = q_props['T2']
                        T1 = q_props['T1']
                        if T2_total > 0.5 * T1: # Ensure T2_prime is positive
                            T2_prime_inv = (1.0 / T2_total) - (0.5 / T1)
                            if T2_prime_inv > 0:
                                T2_prime = 1.0 / T2_prime_inv
                                prob_t2 = 1.0 - math.exp(-duration / T2_prime)
                                if prob_t2 > 1e-12:
                                    noise['incoherent_post_op'].append({
                                        'channel_type': 'phase_damping', 'target_qubits': [q], 'params': {'probability': prob_t2}
                                    })
                                    self._internal_logger.debug(f"HardwareBackend: Added T2 (prob={prob_t2:.4e}) for qubit {q}.")
                    else: # No T1 data, use raw T2 for phase damping
                        prob_t2 = 1.0 - math.exp(-duration / q_props['T2'])
                        if prob_t2 > 1e-12:
                            noise['incoherent_post_op'].append({
                                'channel_type': 'phase_damping', 'target_qubits': [q], 'params': {'probability': prob_t2}
                            })
                            self._internal_logger.debug(f"HardwareBackend: Added T2 (prob={prob_t2:.4e}) for qubit {q} (no T1).")
            
            # --- 步骤 3: 模拟门本身的实现错误 (Depolarizing Noise) ---
            error_rate: float = gate_info.get('error_rate', 0.0)
            if error_rate > 1e-12:
                noise['incoherent_post_op'].append({
                    'channel_type': 'depolarizing', 'target_qubits': qubits, 'params': {'probability': error_rate}
                })
                self._internal_logger.debug(f"HardwareBackend: Added depolarizing error (prob={error_rate:.4e}) for '{op_name}' on {qubits}.")
                
            return noise

        def get_readout_error_matrix(self, num_qubits: int) -> Optional[Dict[int, float]]:
            """
            从校准数据中提取每个量子比特的读出错误率。
            """
            # --- 输入验证 ---
            if not isinstance(num_qubits, int) or num_qubits < 0:
                self._internal_logger.error(f"HardwareBackend: Invalid 'num_qubits' {num_qubits} for readout error matrix. Must be non-negative integer.")
                return None

            readout_errors: Dict[int, float] = {}
            raw_qubits_calib = self.calib.get('qubits', {})

            if not raw_qubits_calib:
                self._internal_logger.debug("HardwareBackend: No qubit calibration data found for readout errors.")
                return None

            for qubit_idx, properties in raw_qubits_calib.items():
                # 这里的 qubit_idx 已经是整数，因为在 _validate_and_preprocess_calibration 中转换了
                if not (0 <= qubit_idx < num_qubits):
                    self._internal_logger.warning(f"HardwareBackend: Readout error for qubit {qubit_idx} is out of bounds for {num_qubits} system, ignoring.")
                    continue
                if 'readout_error' in properties:
                    error_rate = properties['readout_error'] # 已经转换为float
                    if 0.0 <= error_rate <= 1.0:
                        readout_errors[qubit_idx] = error_rate
                        self._internal_logger.debug(f"HardwareBackend: Found readout error {error_rate:.4e} for qubit {qubit_idx}.")
                    else:
                        self._internal_logger.warning(f"HardwareBackend: Invalid readout_error '{error_rate}' for qubit {qubit_idx}, must be in [0.0, 1.0]. Ignoring.")

            return readout_errors if readout_errors else None

    class CorrelatedNoise(NoiseModel):
        """
        一个模拟空间关联噪声（串扰）的增强模型。

        此模型的核心功能是通过一个高度可配置的 `crosstalk_map` 来定义
        当一个多比特门作用时，对目标比特和旁观者比特产生的相干和非相干影响。

        核心特性:
        - 方向性: 能够为 `GATE(A,B)` 和 `GATE(B,A)` 定义不同的噪声。
        - 统一误差模型: 允许用一个完整的、包含相干误差的酉算子（以子电路形式）
          来替换理想的门操作。这可以同时模拟在目标比特和旁观者比特上的串扰。
        - 混合噪声: 支持在相干误差之后应用非相干量子通道。
        - 通配符支持: 允许为作用于特定量子比特上的“任何”多比特门定义通用规则。

        """
        def __init__(self, crosstalk_map: Dict[str, Any]):
            """
            初始化增强的关联噪声模型。

            Args:
                crosstalk_map (Dict[str, Any]):
                    一个描述串扰规则的字典。其结构如下:
                    {
                        'gate_name_or_any': { # e.g., 'cnot', 'cz', or 'any' for wildcard
                            (q1, q2, ...): {  # 精确、有序的量子比特元组
                                'description': '人类可读的描述 (可选)',
                                'error_model': {
                                    'coherent_unitary_replacement': { # (可选)
                                        'circuit': QuantumCircuit(...)
                                    },
                                    'incoherent_post_op': [ # (可选)
                                        {'channel_type': '...', 'target_qubits': [...], 'params': {...}}
                                    ]
                                }
                            }
                        }
                    }

            Raises:
                TypeError: 如果 crosstalk_map 不是字典。
                ValueError: 如果 crosstalk_map 的结构或内容无效。
            """
            super().__init__() # 调用基类初始化，设置logger
            
            # --- 输入验证 ---
            if not isinstance(crosstalk_map, dict):
                raise TypeError("CorrelatedNoise: 'crosstalk_map' must be a dictionary.")
            
            # 执行深度验证和预处理
            self.crosstalk_map = self._validate_and_preprocess_map(crosstalk_map)
            
            self._internal_logger.info(f"CorrelatedNoise initialized, containing {sum(len(rules) for rules in self.crosstalk_map.values())} rules.")

        def _validate_and_preprocess_map(self, raw_map: Dict[str, Any]) -> Dict[str, Any]:
            """
            对输入的 `crosstalk_map` 进行深度验证，确保其结构和类型正确。
            [健壮性改进] 更全面地检查 `QuantumCircuit` 实例，并确保 `target_qubits` 合理。
            """
            validated_map: Dict[str, Any] = {}
            for gate_key, rules_for_gate in raw_map.items():
                if not isinstance(gate_key, str) or not gate_key.strip():
                    raise TypeError(f"CorrelatedNoise: Crosstalk map top-level keys must be non-empty strings (gate name or 'any'), but got {gate_key}.")
                if not isinstance(rules_for_gate, dict):
                    raise ValueError(f"CorrelatedNoise: Rules for gate '{gate_key}' must be a dictionary, but got {type(rules_for_gate).__name__}.")
                
                validated_rules: Dict[Tuple[int, ...], Any] = {}
                for qubits_tuple, details in rules_for_gate.items():
                    # 验证量子比特元组
                    if not isinstance(qubits_tuple, tuple) or not all(isinstance(q, int) and q >= 0 for q in qubits_tuple):
                        raise ValueError(f"CorrelatedNoise: Rule key for gate '{gate_key}' must be a tuple of non-negative integers, but got {qubits_tuple}.")
                    if not qubits_tuple and gate_key != 'any':
                         self._internal_logger.warning(f"CorrelatedNoise: Rule key {qubits_tuple} is empty for gate '{gate_key}'. This rule may not be effective for multi-qubit gates.")

                    # 验证规则详情字典
                    if not isinstance(details, dict) or 'error_model' not in details:
                        raise ValueError(f"CorrelatedNoise: Rule for '({gate_key}, {qubits_tuple})' must be a dictionary containing an 'error_model' key.")
                    
                    error_model = details['error_model']
                    if not isinstance(error_model, dict):
                        raise ValueError(f"CorrelatedNoise: Rule for '({gate_key}, {qubits_tuple})' - 'error_model' must be a dictionary, but got {type(error_model).__name__}.")

                    # 验证 'coherent_unitary_replacement' 部分
                    if 'coherent_unitary_replacement' in error_model:
                        replacement_info = error_model['coherent_unitary_replacement']
                        if not isinstance(replacement_info, dict) or 'circuit' not in replacement_info:
                             raise ValueError(f"CorrelatedNoise: Rule for '({gate_key}, {qubits_tuple})' - 'coherent_unitary_replacement' must be a dictionary containing a 'circuit' key.")
                        
                        circuit = replacement_info['circuit']
                        # [健壮性改进] 确保是 QuantumCircuit 实例
                        if not isinstance(circuit, QuantumCircuit): 
                             raise TypeError(f"CorrelatedNoise: Rule for '({gate_key}, {qubits_tuple})' - 'circuit' must be a QuantumCircuit instance, but got {type(circuit).__name__}.")
                        
                        # [健壮性改进] 检查替换电路的比特数是否与受影响的比特兼容
                        # 确保替换电路的比特数能够容纳所有在 qubits_tuple 中定义的比特。
                        if qubits_tuple: # 如果 qubits_tuple 非空
                            max_involved_qubit = max(qubits_tuple)
                            if circuit.num_qubits <= max_involved_qubit:
                                # 这是一个警告，因为替换电路可能太小无法正确模拟串扰
                                self._internal_logger.warning(
                                    f"CorrelatedNoise: Replacement circuit for '({gate_key}, {qubits_tuple})' has {circuit.num_qubits} qubits, "
                                    f"which is <= max involved qubit index ({max_involved_qubit}). "
                                    "This circuit may not correctly model all effects on these qubits. Consider making the replacement circuit larger."
                                )
                        
                    # 验证 'incoherent_post_op' 部分
                    if 'incoherent_post_op' in error_model:
                        incoherent_ops = error_model['incoherent_post_op']
                        if not isinstance(incoherent_ops, list):
                            raise TypeError(f"CorrelatedNoise: Rule for '({gate_key}, {qubits_tuple})' - 'incoherent_post_op' must be a list, but got {type(incoherent_ops).__name__}.")
                        for op_desc in incoherent_ops:
                            if not isinstance(op_desc, dict) or 'channel_type' not in op_desc or 'params' not in op_desc:
                                raise ValueError(f"CorrelatedNoise: Each incoherent operation in 'incoherent_post_op' must be a dictionary with 'channel_type' and 'params' keys.")
                            
                            # [健壮性改进] 验证 target_qubits
                            channel_target_qubits = op_desc.get('target_qubits')
                            if channel_target_qubits is not None:
                                if not isinstance(channel_target_qubits, (int, list)):
                                    raise TypeError(f"CorrelatedNoise: Incoherent op target_qubits must be an int or list of ints, got {type(channel_target_qubits).__name__}.")
                                if isinstance(channel_target_qubits, list) and not all(isinstance(q, int) and q >= 0 for q in channel_target_qubits):
                                    raise ValueError(f"CorrelatedNoise: Incoherent op target_qubits list must contain non-negative integers.")
                            
                            # 可以在这里进一步验证 channel_type 和 params 的合法性，但通常在 apply_quantum_channel 内部完成。
                    
                    validated_rules[qubits_tuple] = details
                
                validated_map[gate_key] = validated_rules
            return validated_map

        def get_noise_for_op(self, op_name: str, qubits: List[int]) -> Dict[str, Any]:
            """
            根据操作名称和精确的、有序的量子比特列表查找并返回噪声模型。
            
            查找优先级:
            1. 精确匹配: `crosstalk_map['gate_name'][(q1, q2, ...)]`
            2. 通配符匹配: `crosstalk_map['any'][(q1, q2, ...)]`
            """
            # --- 输入验证 ---
            if not isinstance(op_name, str) or not op_name.strip():
                self._internal_logger.error(f"CorrelatedNoise: Invalid 'op_name' '{op_name}'. Must be a non-empty string.")
                return {} # 安全返回空噪声
            if not isinstance(qubits, list) or not all(isinstance(q, int) for q in qubits):
                self._internal_logger.error(f"CorrelatedNoise: Invalid 'qubits' {qubits}. Must be a list of integers.")
                return {} # 安全返回空噪声

            qubits_tuple = tuple(qubits)
            
            # --- 步骤 1: 尝试精确匹配 (e.g., 'cnot' on (0, 1)) ---
            if op_name in self.crosstalk_map:
                gate_specific_rules = self.crosstalk_map[op_name]
                if qubits_tuple in gate_specific_rules:
                    self._internal_logger.debug(f"CorrelatedNoise: Found exact rule for '{op_name}' on {qubits_tuple}.")
                    return self._extract_error_model(gate_specific_rules[qubits_tuple])

            # --- 步骤 2: 尝试通配符匹配 (e.g., 'any' on (0, 1)) ---
            if 'any' in self.crosstalk_map:
                any_gate_rules = self.crosstalk_map['any']
                if qubits_tuple in any_gate_rules:
                    self._internal_logger.debug(f"CorrelatedNoise: Found wildcard rule for 'any' on {qubits_tuple} (applied to '{op_name}').")
                    return self._extract_error_model(any_gate_rules[qubits_tuple])

            # --- 步骤 3: 如果没有找到任何匹配，返回空字典 ---
            self._internal_logger.debug(f"CorrelatedNoise: No matching rule found for '{op_name}' on {qubits_tuple}. No noise applied.")
            return {}

        def _extract_error_model(self, rule_details: Dict[str, Any]) -> Dict[str, Any]:
            """
            [v1.5.11 data structure fix] 从规则详情字典中安全地提取并构建返回给执行引擎的噪声字典。
            此版本修复了 coherent_unitary_replacement 返回值结构不符合接口定义的问题。
            """
            error_model = rule_details.get('error_model', {})
            noise_to_apply: Dict[str, Any] = {}

            # a) 提取相干误差替换电路
            if 'coherent_unitary_replacement' in error_model:
                replacement_info = error_model['coherent_unitary_replacement']
                if 'circuit' in replacement_info and isinstance(replacement_info['circuit'], QuantumCircuit):
                    # --- [核心修复] ---
                    # 返回的结构必须是一个包含 'circuit' 键的字典，以符合 get_noise_for_op 的接口定义。
                    noise_to_apply['coherent_unitary_replacement'] = {
                        'circuit': copy.deepcopy(replacement_info['circuit'])
                    }
                    self._internal_logger.debug(f"Extracted coherent_unitary_replacement circuit: {replacement_info['circuit'].description}.")
                    # --- [修复结束] ---
                else:
                    self._internal_logger.warning("CorrelatedNoise: coherent_unitary_replacement found but 'circuit' key is missing or its value is invalid. Ignoring.")
            
            # b) 提取非相干后置操作
            if 'incoherent_post_op' in error_model:
                if isinstance(error_model['incoherent_post_op'], list):
                    # [健壮性改进] 深拷贝列表及其内容，防止意外修改原始配置
                    noise_to_apply['incoherent_post_op'] = copy.deepcopy(error_model['incoherent_post_op'])
                    self._internal_logger.debug(f"Extracted {len(noise_to_apply['incoherent_post_op'])} incoherent_post_op channels.")
                else:
                    self._internal_logger.warning("CorrelatedNoise: incoherent_post_op found but not a list. Ignoring.")

            return noise_to_apply


# ========================================================================
# --- 8. [扩展] 算法构建器 ---
# ========================================================================




class AlgorithmBuilders:
    """
    [健壮性改进版] 一个命名空间类，用于组织所有高级算法的线路构建函数。

    此类本身不应该被实例化。它的所有方法都应该是 `@staticmethod`，
    因为它们是无状态的辅助函数，接收参数并返回一个 `QuantumCircuit` 对象。

    [终极统一版] 此类现在还包含一个宏注册表 (`_macro_definitions`) 和
    相关的管理方法 (`register_macro`, `get_macro_definition`)。这使得
    高级门（如 Toffoli）的分解定义可以被集中管理，成为全系统的“单一事实来源”，
    从而解决了不同模块间实现不一致的问题。

    [新增改进] 所有门（包括原子门）都通过宏注册表管理。
    """
    _macro_definitions: Dict[str, Tuple[Callable[..., Any], int, Optional[str]]] = {} # 存储 (func, num_qubits, description)
    _internal_logger: logging.Logger = logging.getLogger(f"{__name__}.AlgorithmBuilders") # 直接使用类名

    @staticmethod
    def register_macro(name: str, num_qubits: int, description: Optional[str] = None):
        """
        一个装饰器，用于将一个函数注册为高级指令的官方“宏”定义。
        
        Args:
            name (str): 宏指令的名称，例如 'toffoli'。
            num_qubits (int): 此宏操作的量子比特数。
                              对于固定比特数门 (H, CNOT)，这是确切的比特数。
                              对于可变比特数门 (MCX, MCZ)，设为 0。
            description (Optional[str]): 宏的简要描述。

        Raises:
            ValueError: 如果 `name` 无效或 `num_qubits` 为负数。
            TypeError: 如果 `name` 或 `description` 类型不正确。
        """
        # [健壮性改进] 输入验证
        if not isinstance(name, str) or not name.strip():
            AlgorithmBuilders._internal_logger.error(f"register_macro: 'name' must be a non-empty string, but got '{name}'.")
            raise ValueError("Macro name must be a non-empty string.")
        if not isinstance(num_qubits, int) or num_qubits < 0:
            AlgorithmBuilders._internal_logger.error(f"register_macro: 'num_qubits' must be a non-negative integer, but got {num_qubits}.")
            raise ValueError("Macro num_qubits must be a non-negative integer.")
        if description is not None and not isinstance(description, str):
            AlgorithmBuilders._internal_logger.error(f"register_macro: 'description' must be a string or None, but got {type(description).__name__}.")
            raise TypeError("Macro description must be a string or None.")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in AlgorithmBuilders._macro_definitions:
                AlgorithmBuilders._internal_logger.warning(f"Macro '{name}' is being re-registered. Overwriting existing definition.")
            AlgorithmBuilders._macro_definitions[name] = (func, num_qubits, description)
            AlgorithmBuilders._internal_logger.debug(f"Macro '{name}' (N={num_qubits}) registered.")
            return func
        return decorator

    @staticmethod
    def get_macro_definition(name: str) -> Optional[Tuple[Callable[..., Any], int, Optional[str]]]:
        """
        根据指令名称查询其官方的宏定义（分解函数）。
        
        Args:
            name (str): 宏指令的名称。

        Returns:
            Optional[Tuple[Callable[..., Any], int, Optional[str]]]:
                如果找到，返回一个包含 (分解函数, 比特数, 描述) 的元组；
                否则返回 None。

        Raises:
            TypeError: 如果 `name` 不是字符串。
        """
        # [健壮性改进] 输入验证
        if not isinstance(name, str):
            AlgorithmBuilders._internal_logger.error(f"get_macro_definition: 'name' must be a string, but got {type(name).__name__}.")
            raise TypeError("Macro name must be a string.")
        return AlgorithmBuilders._macro_definitions.get(name)

    @staticmethod
    @lru_cache(maxsize=128) # 缓存宏的酉矩阵，避免重复计算
    def _get_macro_unitary(macro_name: str, args_tuple: Tuple[Any, ...]) -> Any:
        """
        [健壮性改进版] [v1.5.12 qubit index mapping fix]
        自动计算指定宏定义（包括基础门）的有效酉矩阵。
        此版本修复了在计算局部酉矩阵时，因错误传递全局比特索引而导致的 ValueError。

        修复逻辑:
        在调用宏函数之前，将参数中的所有 qubit 索引（整数）提取出来，
        并创建一个从全局索引到局部索引（0, 1, ...）的映射。然后使用
        这个映射来转换参数，确保传递给宏函数的是局部索引。

        Args:
            macro_name (str): 宏的名称。
            args_tuple (Tuple[Any, ...]): 宏的位置参数（不包括 circuit 实例本身）。
                                          例如，对于 RX(q, theta)，args_tuple = (q, theta)。

        Returns:
            Any: 宏的酉矩阵，类型与当前后端兼容。
                 (对于单比特门，是 2x2；对于双比特门，是 4x4，等等)

        Raises:
            ValueError: 如果宏未注册，参数不匹配或无法计算其酉矩阵。
            TypeError: 如果 `macro_name` 或 `args_tuple` 类型不正确。
            RuntimeError: 如果底层 `get_effective_unitary` 失败。
        """
        # [健壮性改进] 输入验证
        if not isinstance(macro_name, str):
            AlgorithmBuilders._internal_logger.error(f"_get_macro_unitary: 'macro_name' must be a string, but got {type(macro_name).__name__}.")
            raise TypeError("Macro name must be a string.")
        if not isinstance(args_tuple, tuple):
            AlgorithmBuilders._internal_logger.error(f"_get_macro_unitary: 'args_tuple' must be a tuple, but got {type(args_tuple).__name__}.")
            raise TypeError("Arguments tuple must be a tuple.")

        macro_info = AlgorithmBuilders.get_macro_definition(macro_name)
        if macro_info is None:
            AlgorithmBuilders._internal_logger.error(f"_get_macro_unitary: Macro '{macro_name}' is not registered.")
            raise ValueError(f"Macro '{macro_name}' is not registered.")
        
        macro_func, num_local_qubits_in_macro_def, _ = macro_info

        # 确定用于生成酉矩阵的临时电路的比特数
        actual_macro_num_qubits: int
        mapped_args_for_macro_func: Tuple[Any, ...]

        # --- [核心修复] ---
        # 提取所有全局 qubit 索引，并创建到局部索引 (0, 1, ...) 的映射
        all_global_qubits_in_args = sorted(list(set(
            q for arg in args_tuple 
            for q in (arg if isinstance(arg, list) else [arg]) 
            if isinstance(q, int)
        )))

        if num_local_qubits_in_macro_def > 0: # 固定比特数宏
            if len(all_global_qubits_in_args) != num_local_qubits_in_macro_def:
                AlgorithmBuilders._internal_logger.warning(
                    f"Number of distinct qubit indices ({len(all_global_qubits_in_args)}) in args {args_tuple} "
                    f"does not match macro definition's expected qubit count ({num_local_qubits_in_macro_def}) for '{macro_name}'. "
                    "This can happen for parameterized gates. Proceeding with macro definition's qubit count."
                )
            actual_macro_num_qubits = num_local_qubits_in_macro_def
        
        elif num_local_qubits_in_macro_def == 0: # 可变比特数宏
            actual_macro_num_qubits = len(all_global_qubits_in_args)
            if actual_macro_num_qubits == 0 and any(isinstance(arg, list) for arg in args_tuple):
                # 特殊情况： controls 列表可能为空，导致 all_global_qubits_in_args 也为空
                try:
                    controls = next(arg for arg in args_tuple if isinstance(arg, list))
                    if not controls: # 如果是空列表
                        target = next(arg for arg in args_tuple if isinstance(arg, int))
                        actual_macro_num_qubits = 1 # 至少需要一个比特
                except StopIteration:
                    pass

        # 创建从全局索引到局部索引的映射
        global_to_local_map = {q_global: q_local for q_local, q_global in enumerate(all_global_qubits_in_args)}

        # 转换参数，将所有全局 qubit 索引替换为局部索引
        mapped_args_list = []
        for arg in args_tuple:
            if isinstance(arg, int) and arg in global_to_local_map:
                mapped_args_list.append(global_to_local_map[arg])
            elif isinstance(arg, list) and all(isinstance(q, int) for q in arg):
                mapped_args_list.append([global_to_local_map.get(q, q) for q in arg])
            else:
                mapped_args_list.append(arg)
        mapped_args_for_macro_func = tuple(mapped_args_list)
        # --- [修复结束] ---

        # 创建一个临时的 QuantumCircuit 来容纳宏展开的结果
        if _get_effective_unitary_placeholder is None or not isinstance(_get_effective_unitary_placeholder, Callable):
            AlgorithmBuilders._internal_logger.critical("_get_macro_unitary: 'get_effective_unitary' function is not yet defined or properly set up in quantum_core.")
            raise RuntimeError("get_effective_unitary function is not yet defined or properly set up.")

        temp_circuit = QuantumCircuit(actual_macro_num_qubits, description=f"Unitary derivation for {macro_name}")
        
        try:
            # 调用宏函数，将分解后的指令添加到 temp_circuit
            macro_func(temp_circuit, *mapped_args_for_macro_func, **{})
            
            # 使用 get_effective_unitary 计算这个临时电路的等效酉矩阵
            unitary = _get_effective_unitary_placeholder(temp_circuit)
            return unitary
        except Exception as e:
            AlgorithmBuilders._internal_logger.error(f"_get_macro_unitary: Failed to derive unitary for macro '{macro_name}' with args {args_tuple}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to derive unitary for macro '{macro_name}' with args {args_tuple}.") from e

    @staticmethod
    def build_qft_circuit(num_qubits: int, inverse: bool = False, do_swaps: bool = True) -> 'QuantumCircuit':
        """
        [最终正确性修复版 v3 - 小端序标准版] 构建一个量子傅里叶变换 (QFT) 或其逆 (IQFT) 的线路。
        此版本严格遵循标准的小端序约定 (|q_{n-1}...q_0>)，与所有内核函数保持一致。
        """
        if not isinstance(num_qubits, int) or num_qubits < 0:
            raise ValueError("num_qubits must be a non-negative integer for QFT circuit.")
        if not isinstance(inverse, bool):
            raise TypeError("Inverse must be a boolean.")
        if not isinstance(do_swaps, bool):
            raise TypeError("do_swaps must be a boolean.")

        circuit_name = f"{num_qubits}-qubit {'Inverse ' if inverse else ''}QFT"
        qc_build = QuantumCircuit(num_qubits=num_qubits, description=circuit_name)
        
        if num_qubits == 0:
            return qc_build

        if not inverse:
            # --- 正向 QFT (小端序) ---
            # 从最高位比特 (n-1) 到最低位比特 (0)
            for i in range(num_qubits - 1, -1, -1):
                qc_build.h(i)
                for j in range(i - 1, -1, -1):
                    angle = math.pi / (2**(i - j))
                    qc_build.cp(j, i, angle)
            
            if do_swaps:
                for i in range(num_qubits // 2):
                    qc_build.swap(i, num_qubits - 1 - i)
        else:
            # --- 逆向 QFT (IQFT) (小端序) ---
            if do_swaps:
                for i in range(num_qubits // 2):
                    qc_build.swap(i, num_qubits - 1 - i)
            
            # 从最低位比特 (0) 到最高位比特 (n-1)
            for i in range(num_qubits):
                for j in range(i):
                    angle = -math.pi / (2**(i - j))
                    qc_build.cp(j, i, angle)
                qc_build.h(i)
                
        return qc_build
    
    
    @staticmethod
    def build_grover_oracle_circuit(num_qubits: int, target_state_int: int) -> 'QuantumCircuit':
        """
        [健壮性改进版] 为Grover搜索算法构建一个标记单个目标状态的“神谕”(Oracle)线路。

        Args:
            num_qubits (int): 量子比特数量。
            target_state_int (int): 要标记的目标状态的整数表示。

        Returns:
            QuantumCircuit: 构建的 Grover 神谕线路。

        Raises:
            ValueError: 如果 `num_qubits` 为非正数或 `target_state_int` 超出范围。
            TypeError: 如果 `num_qubits` 或 `target_state_int` 类型不正确。
        """
        # [健壮性改进] 输入验证
        if not isinstance(num_qubits, int) or num_qubits < 1:
            AlgorithmBuilders._internal_logger.error(f"build_grover_oracle_circuit: 'num_qubits' must be a positive integer, got {num_qubits}.")
            raise ValueError("num_qubits must be a positive integer for the Grover oracle.")
        if not isinstance(target_state_int, int) or not (0 <= target_state_int < (1 << num_qubits)):
            AlgorithmBuilders._internal_logger.error(f"build_grover_oracle_circuit: Target state {target_state_int} is out of range for {num_qubits} qubits.")
            raise ValueError(f"Target state {target_state_int} is out of range for {num_qubits} qubits.")
            
        # 修正: 直接使用 QuantumCircuit 类名
        qc_build = QuantumCircuit(num_qubits=num_qubits, description=f"Oracle for state |{target_state_int}⟩ ({target_state_int:0{num_qubits}b}b)")
        
        # 翻转比特以将目标态变为 |0...0>
        qubits_to_flip = [i for i in range(num_qubits) if not ((target_state_int >> i) & 1)]
        
        if qubits_to_flip:
            qc_build.barrier(*qubits_to_flip) # 调用 BARRIER 宏 (或直接使用 qc_build._add_raw_gate("barrier", ...))
            for i in qubits_to_flip:
                qc_build.x(i) # 调用 X 宏
            qc_build.barrier(*qubits_to_flip)

        if num_qubits > 0:
            target_qubit_for_oracle = num_qubits - 1 # 通常使用最高位作为目标比特
            control_qubits = list(range(num_qubits - 1)) # 其他所有比特作为控制比特
            
            qc_build.h(target_qubit_for_oracle) # 调用 H 宏
            
            if control_qubits: # 如果有控制比特
                qc_build.mcx(control_qubits, target_qubit_for_oracle) # 调用 MCX 宏
            else: # 只有1个比特时，MCX([], target) == Z(target)，这里逻辑上等价于Z门对|1>态作用，但 oracle 的构建方式不同
                qc_build.z(target_qubit_for_oracle) # 对于单比特 oracle |1>， Z 门即可实现相位翻转

            qc_build.h(target_qubit_for_oracle) # 调用 H 宏
        
        # 翻转比特以恢复目标态
        if qubits_to_flip:
            qc_build.barrier(*qubits_to_flip)
            for i in qubits_to_flip:
                qc_build.x(i)
            qc_build.barrier(*qubits_to_flip)
                
        return qc_build
    
    
    
    
    @staticmethod
    def build_grover_diffusion_circuit(num_qubits: int) -> 'QuantumCircuit':
        """
        [健壮性改进版] 构建Grover扩散算子线路。
        该算子的作用是围绕平均振幅进行反射，其标准分解为 H-X-MCZ-X-H。

        Args:
            num_qubits (int): 量子比特数量。

        Returns:
            QuantumCircuit: 构建的 Grover 扩散算子线路。

        Raises:
            ValueError: 如果 `num_qubits` 为非正数。
            TypeError: 如果 `num_qubits` 类型不正确。
        """
        # [健壮性改进] 输入验证
        if not isinstance(num_qubits, int) or num_qubits < 1:
            AlgorithmBuilders._internal_logger.error(f"build_grover_diffusion_circuit: 'num_qubits' must be a positive integer, got {num_qubits}.")
            raise ValueError("num_qubits must be a positive integer for the Grover diffusion operator.")
        
        # 修正: 直接使用 QuantumCircuit 类名
        qc_build = QuantumCircuit(num_qubits=num_qubits, description="Grover Diffusion Operator")
        
        # 1. 对所有比特应用Hadamard
        for i in range(num_qubits):
            qc_build.h(i)
        
        # 2. 对所有比特应用X
        for i in range(num_qubits):
            qc_build.x(i)
            
        # 3. 应用多控制Z门 (MCZ)
        # MCZ 作用在所有比特上。一种实现方式是 H-MCX-H。
        # 这里我们直接调用 MCZ 宏，它更基础。
        if num_qubits > 1:
            controls = list(range(num_qubits - 1))
            target = num_qubits - 1
            qc_build.mcz(controls, target)
        elif num_qubits == 1:
            # 对于单个比特，扩散算子就是 Z 门
            qc_build.z(0)

        # 4. 对所有比特应用X
        for i in range(num_qubits):
            qc_build.x(i)
        
        # 5. 对所有比特应用Hadamard
        for i in range(num_qubits):
            qc_build.h(i)
        
        return qc_build
    @staticmethod
    def build_trotter_step_circuit(hamiltonian: 'Hamiltonian', time_step: float) -> 'QuantumCircuit':
        """
        [健壮性改进版] 构建一个一阶 Suzuki-Trotter 分解的单步量子线路。

        Args:
            hamiltonian (Hamiltonian): 哈密顿量，一个 `List[PauliString]` 对象。
            time_step (float): 时间步长 `dt`。

        Returns:
            QuantumCircuit: 构建的 Trotter 步线路。

        Raises:
            TypeError: 如果 `hamiltonian` 或 `time_step` 类型不正确。
            ValueError: 如果 `hamiltonian` 非厄米，或包含不支持的 Pauli 字符串。
        """
        # [健壮性改进] 输入验证
        if not isinstance(hamiltonian, list) or (hamiltonian and not all(isinstance(ps, PauliString) for ps in hamiltonian)):
            AlgorithmBuilders._internal_logger.error(f"build_trotter_step_circuit: 'hamiltonian' must be a list of PauliString objects, got {type(hamiltonian).__name__}.")
            raise TypeError("Hamiltonian must be a list of PauliString objects.")
        if not isinstance(time_step, (float, int)):
            AlgorithmBuilders._internal_logger.error(f"build_trotter_step_circuit: 'time_step' must be a numeric value, got {type(time_step).__name__}.")
            raise TypeError("time_step must be a numeric value.")
        
        if math.isclose(float(time_step), 0.0, abs_tol=1e-12):
            return QuantumCircuit(0, description="Trotter Step for dt=0 (Identity)")

        max_qubit_idx = -1
        for ps_index, ps in enumerate(hamiltonian):
            # [健壮性改进] 检查哈密顿量是否厄米 (系数必须为实数)
            if abs(ps.coefficient.imag) > 1e-9:
                AlgorithmBuilders._internal_logger.error(f"build_trotter_step_circuit: Hamiltonian term {ps_index} ('{ps}') has a complex coefficient ({ps.coefficient}). Hamiltonian must be Hermitian (real coefficients).")
                raise ValueError(f"Hamiltonian must be Hermitian (real coefficients), but found complex coefficient {ps.coefficient} in term '{ps}'.")
            if ps.pauli_map:
                max_qubit_idx = max(max_qubit_idx, max(ps.pauli_map.keys()))
        num_qubits = max_qubit_idx + 1 if max_qubit_idx >= 0 else 0

        # [健壮性改进] 如果 num_qubits=0 且有非零时间步，返回一个 0 比特线路
        if num_qubits == 0 and abs(float(time_step)) > 1e-12:
            if any(len(ps.pauli_map) == 0 and abs(ps.coefficient) > 1e-12 for ps in hamiltonian):
                 AlgorithmBuilders._internal_logger.warning("build_trotter_step_circuit: Hamiltonian contains non-zero identity terms but no qubits. This will result in a global phase only.")
            return QuantumCircuit(0, description=f"Trotter Step (0-qubit, dt={time_step:.4f})")
        
        # 修正: 直接使用 QuantumCircuit 类名
        qc_build = QuantumCircuit(num_qubits=num_qubits, description=f"Trotter Step for dt={time_step:.4f}")

        for ps_index, pauli_string in enumerate(hamiltonian):
            coefficient = pauli_string.coefficient.real # 已经验证为实数
            rotation_angle = 2 * coefficient * time_step

            if math.isclose(float(rotation_angle), 0.0, abs_tol=1e-12):
                AlgorithmBuilders._internal_logger.debug(f"build_trotter_step_circuit: Skipping Hamiltonian term {ps_index} ('{pauli_string}') as rotation angle is negligible.")
                continue
            
            non_identity_ops = {q: op for q, op in pauli_string.pauli_map.items() if op != 'I'}
            
            # [健壮性改进] 处理纯全局相位项 (如 1.0 * I)
            if not non_identity_ops:
                # 纯单位矩阵项只引入全局相位，但这里只构建门线路，全局相位由 get_effective_unitary 处理
                # 单个门的线路不体现全局相位，跳过
                AlgorithmBuilders._internal_logger.debug(f"build_trotter_step_circuit: Skipping pure identity term {ps_index} ('{pauli_string}').")
                continue

            qubit_indices = sorted(non_identity_ops.keys())
            
            # 针对不同数量的非单位 Pauli 操作进行分解
            if len(qubit_indices) == 1:
                q = qubit_indices[0]
                op = non_identity_ops[q]
                if op == 'X': qc_build.rx(q, rotation_angle) # 调用 RX 宏
                elif op == 'Y': qc_build.ry(q, rotation_angle) # 调用 RY 宏
                elif op == 'Z': qc_build.rz(q, rotation_angle) # 调用 RZ 宏
                else:
                    AlgorithmBuilders._internal_logger.error(f"build_trotter_step_circuit: Unsupported single-qubit Pauli operator '{op}' in term '{pauli_string}'.")
                    raise ValueError(f"Unsupported single-qubit Pauli operator '{op}' in term '{pauli_string}'.")
            elif len(qubit_indices) == 2:
                q1, q2 = qubit_indices[0], qubit_indices[1]
                op1, op2 = non_identity_ops[q1], non_identity_ops[q2]
                
                # [健壮性改进] 支持 RXX, RYY, RZZ
                if op1 == 'X' and op2 == 'X': qc_build.rxx(q1, q2, rotation_angle) # 调用 RXX 宏
                elif op1 == 'Y' and op2 == 'Y': qc_build.ryy(q1, q2, rotation_angle) # 调用 RYY 宏
                elif op1 == 'Z' and op2 == 'Z': qc_build.rzz(q1, q2, rotation_angle) # 调用 RZZ 宏
                else:
                    AlgorithmBuilders._internal_logger.warning(f"build_trotter_step_circuit: Encountered non-diagonal 2-qubit Pauli string '{op1}{op2}' on ({q1},{q2}) in term '{pauli_string}'. Using standard decomposition (requires H/CNOTs).")
                    # 更复杂的分解方法，例如 for XX, YY, ZZ when not on specific qubits.
                    # Qiskit的通用分解：U_xy(theta) = Rz(-pi/2) Ry(pi/2) Rz(pi/2) CX(0,1) Rz(-theta/2) CX(0,1) Rz(-pi/2) Ry(-pi/2) Rz(pi/2)
                    # For simplicity, if RXX, RYY, RZZ macros are available, use them.
                    # Otherwise, a generic decomposition of exp(-i * angle * PauliString) is complex.
                    # 对于非对称的 Pauli 串，需要进行基变换和 CNOT 链
                    # 这里假设我们只支持 XX, YY, ZZ (已经通过 rxx, ryy, rzz 宏实现)
                    # 如果有不支持的2比特，则继续抛出 ValueError
                    raise ValueError(f"Unsupported 2-qubit Pauli string for Trotter decomposition: '{op1}{op2}' in term '{pauli_string}'.")

            else: # 大于2个比特的 Pauli 字符串，需要使用辅助比特进行分解
                # 这部分需要更高级的分解技术 (例如使用辅助比特实现多控制门，然后转换为RZ)
                # 目前的代码只支持最多2个非单位 Pauli 算子。
                # 这是一个当前版本的限制
                AlgorithmBuilders._internal_logger.error(f"build_trotter_step_circuit: Unsupported Pauli string for Trotter decomposition (more than 2 non-identity qubits): {pauli_string}.")
                raise ValueError(f"Unsupported Pauli string for Trotter decomposition (more than 2 non-identity qubits): {pauli_string}")
                
        return qc_build
    
    
    
    @staticmethod
    def build_qpe_circuit(
        counting_qubits: List[int],
        target_qubits: List[int],
        unitary_circuit: 'QuantumCircuit'
    ) -> 'QuantumCircuit':
        """
        [最终正确性修复版 v4 - 比特序完全统一] 构建量子相位估计算法 (QPE) 的线路。
        此版本修正了受控酉门层的相位编码顺序，以匹配无SWAP的IQFT的解码约定。
        """
        log_prefix = "AlgorithmBuilders.build_qpe_circuit"
        AlgorithmBuilders._internal_logger.debug(f"[{log_prefix}] Starting QPE circuit construction...")
        
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(counting_qubits, list) or not all(isinstance(q, int) and q >= 0 for q in counting_qubits):
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] 'counting_qubits' must be a non-empty list of non-negative integers, got {counting_qubits}.")
            raise ValueError("counting_qubits must be a non-empty list of non-negative integers.")
        if not counting_qubits:
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] 'counting_qubits' list cannot be empty.")
            raise ValueError("'counting_qubits' list cannot be empty.")
        
        if not isinstance(target_qubits, list) or not all(isinstance(q, int) and q >= 0 for q in target_qubits):
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] 'target_qubits' must be a non-empty list of non-negative integers, got {target_qubits}.")
            raise ValueError("target_qubits must be a non-empty list of non-negative integers.")
        if not target_qubits:
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] 'target_qubits' list cannot be empty.")
            raise ValueError("'target_qubits' list cannot be empty.")
            
        if not isinstance(unitary_circuit, QuantumCircuit): 
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] 'unitary_circuit' must be a QuantumCircuit instance, but got {type(unitary_circuit).__name__}.")
            raise TypeError("unitary_circuit must be a QuantumCircuit instance.")
        if unitary_circuit.num_qubits != len(target_qubits):
            AlgorithmBuilders._internal_logger.error(
                f"[{log_prefix}] The number of qubits in 'unitary_circuit' ({unitary_circuit.num_qubits}) "
                f"must match the number of 'target_qubits' ({len(target_qubits)})."
            )
            raise ValueError(
                f"The number of qubits in unitary_circuit ({unitary_circuit.num_qubits}) "
                f"must match the number of target_qubits ({len(target_qubits)})."
            )
        
        # 检查所有量子比特是否唯一且有效
        all_qubits = set(counting_qubits) | set(target_qubits)
        if len(all_qubits) != len(counting_qubits) + len(target_qubits):
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] 'counting_qubits' and 'target_qubits' lists contain overlapping qubits.")
            raise ValueError("Counting qubits and target qubits lists must not overlap.")

        max_global_qubit_idx = -1
        if all_qubits:
            max_global_qubit_idx = max(all_qubits)
        num_qubits = max_global_qubit_idx + 1 if max_global_qubit_idx >= 0 else 0

        qc_build = QuantumCircuit(num_qubits, description=f"QPE ({len(counting_qubits)} counting qubits)")

        # --- 步骤 2: 对计数比特应用 Hadamard 门 ---
        AlgorithmBuilders._internal_logger.debug(f"[{log_prefix}] Adding Hadamard layer to counting qubits.")
        for q in counting_qubits:
            qc_build.h(q)

        # --- 步骤 3: [最终核心修复] 应用受控酉门，使用反向幂次来匹配IQFT(no swaps)的期望输入 ---
        num_counting_qubits = len(counting_qubits)
        for i in range(num_counting_qubits):
            control_qubit = counting_qubits[i]
            
            # 关键修复：counting_qubits[i] (第i个比特) 控制 U 的 2^(n-1-i) 次幂
            # e.g., for n=3: c_0 controls U^4, c_1 controls U^2, c_2 controls U^1
            # 这会在傅里叶基中创建反向编码的相位
            num_applications = 1 << (num_counting_qubits - 1 - i)
            
            AlgorithmBuilders._internal_logger.debug(f"  - Applying U^{num_applications} controlled by qubit {control_qubit}.")

            for _ in range(num_applications):
                for instr in unitary_circuit.instructions:
                    gate_name = instr[0]
                    op_kwargs = instr[-1] if isinstance(instr[-1], dict) else {}
                    op_args = list(instr[1:-1]) if isinstance(instr[-1], dict) else list(instr[1:])

                    # 将局部比特索引映射到全局目标比特索引
                    mapped_args: List[Any] = []
                    for arg in op_args:
                        if isinstance(arg, int):
                            if arg < len(target_qubits):
                                mapped_args.append(target_qubits[arg])
                            else:
                                raise ValueError(f"Qubit index {arg} in unitary_circuit is out of bounds for target_qubits list.")
                        elif isinstance(arg, list) and all(isinstance(idx, int) for idx in arg):
                            mapped_args.append([target_qubits[idx] for idx in arg])
                        else:
                            mapped_args.append(arg)
                    
                    # 将门转换为受控版本并添加到主电路
                    if gate_name == 'x': qc_build.cnot(control_qubit, mapped_args[0], **op_kwargs)
                    elif gate_name in ['rx', 'ry', 'rz', 'p_gate']:
                        controlled_gate_name = 'c' + gate_name if gate_name != 'p_gate' else 'cp'
                        getattr(qc_build, controlled_gate_name)(control_qubit, *mapped_args, **op_kwargs)
                    elif gate_name == 'cnot': qc_build.toffoli(control_qubit, mapped_args[0], mapped_args[1], **op_kwargs)
                    elif gate_name.startswith('mc'):
                        controls_list = mapped_args[0]
                        new_controls = [control_qubit] + controls_list
                        if gate_name == 'mcx': qc_build.mcx(new_controls, mapped_args[1], **op_kwargs)
                        elif gate_name == 'mcz': qc_build.mcz(new_controls, mapped_args[1], **op_kwargs)
                        elif gate_name == 'mcp': qc_build.mcp(new_controls, mapped_args[1], mapped_args[2], **op_kwargs)
                        elif gate_name == 'mcu': qc_build.mcu(new_controls, mapped_args[1], mapped_args[2], **op_kwargs)
                    else:
                        AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] Automatic controlled version for gate '{gate_name}' is not yet implemented in the QPE builder.")
                        raise NotImplementedError(f"Automatic controlled version for gate '{gate_name}' is not implemented in the QPE builder.")


        # --- 步骤 4: 应用不带SWAP的IQFT ---
        AlgorithmBuilders._internal_logger.debug(f"[{log_prefix}] Adding Inverse QFT layer to counting qubits (without SWAPs).")
        
        iqft_circuit = AlgorithmBuilders.build_qft_circuit(num_counting_qubits, inverse=True, do_swaps=False)
        
        # 将IQFT的局部比特映射到全局计数比特
        for instr in iqft_circuit.instructions:
            gate_name = instr[0]
            op_kwargs_iqft = instr[-1] if isinstance(instr[-1], dict) else {}
            op_args_iqft = list(instr[1:-1]) if isinstance(instr[-1], dict) else list(instr[1:])
            
            mapped_args_iqft: List[Any] = []
            for arg in op_args_iqft:
                if isinstance(arg, int):
                    if arg < num_counting_qubits:
                        mapped_args_iqft.append(counting_qubits[arg])
                    else:
                        raise ValueError(f"Qubit index {arg} in IQFT is out of bounds for counting_qubits list.")
                elif isinstance(arg, list) and all(isinstance(i, int) for i in arg):
                    mapped_args_iqft.append([counting_qubits[i] for i in arg])
                else:
                    mapped_args_iqft.append(arg)

            qc_build.add_gate(gate_name, *mapped_args_iqft, **op_kwargs_iqft)

        AlgorithmBuilders._internal_logger.info(f"[{log_prefix}] QPE circuit with {len(qc_build.instructions)} gates constructed successfully.")
        return qc_build
    
    @staticmethod
    def build_hardware_efficient_ansatz(
        num_qubits: int,
        depth: int,
        parameters: List[float],
        entanglement_type: Literal['linear', 'circular', 'full'] = 'linear'
    ) -> 'QuantumCircuit':
        """
        [健 robuste性改进版] 构建一个硬件高效拟设 (Hardware-Efficient Ansatz) 的线路。

        这种拟设结构在变分量子算法 (VQA) 中非常常用，因为它由易于在当前量子硬件上
        实现的单比特旋转门和双比特纠缠门交替组成。

        结构如下，重复 `depth` 次：
        1. **旋转层**: 在每个量子比特上应用参数化的单比特旋转门 (通常是 Ry 和 Rz)。
        2. **纠缠层**: 应用一系列固定的双比特纠缠门 (通常是 CNOT)，其模式由 `entanglement_type` 决定。

        Args:
            num_qubits (int): 电路中的量子比特数量。
            depth (int): 拟设的深度（层数）。每一层包含一个旋转层和一个纠缠层。
            parameters (List[float]): 拟设的参数列表。期望 `depth * num_qubits * 2` 个参数，
                                      因为每个量子比特在每一层都需要两个旋转角度 (Ry 和 Rz)。
            entanglement_type (Literal['linear', 'circular', 'full'], optional):
                纠缠层的 CNOT 连接模式。
                - 'linear': CNOTs 连接相邻的比特 (q_i, q_i+1)。
                - 'circular': 线性连接，并在最后一个和第一个比特之间添加一个 CNOT。
                - 'full': 在所有可能的比特对之间应用 CNOT。
                默认为 'linear'。

        Returns:
            QuantumCircuit: 构建的硬件高效拟设线路。

        Raises:
            ValueError: 如果 `num_qubits`, `depth` 为负数，`parameters` 长度不匹配，或 `entanglement_type` 无效。
            TypeError: 如果输入参数类型不正确。
        """
        log_prefix = "AlgorithmBuilders.build_hardware_efficient_ansatz"
        AlgorithmBuilders._internal_logger.debug(f"[{log_prefix}] Starting ansatz construction...")
        
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(num_qubits, int) or num_qubits < 0:
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] 'num_qubits' must be a non-negative integer, got {num_qubits}.")
            raise ValueError("num_qubits must be a non-negative integer.")
        if not isinstance(depth, int) or depth < 0:
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] 'depth' must be a non-negative integer, got {depth}.")
            raise ValueError("depth must be a non-negative integer.")
        if not isinstance(parameters, list) or not all(isinstance(p, (float, int)) for p in parameters):
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] 'parameters' must be a list of numeric values, got {type(parameters).__name__}.")
            raise TypeError("parameters must be a list of numeric values.")

        expected_params = depth * num_qubits * 2 if num_qubits > 0 else 0
        if len(parameters) != expected_params:
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] Expected {expected_params} parameters for ansatz (depth={depth}, qubits={num_qubits}), but got {len(parameters)}.")
            raise ValueError(f"Expected {expected_params} parameters for ansatz, but got {len(parameters)}.")

        if entanglement_type not in ['linear', 'circular', 'full']:
            AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] Unknown 'entanglement_type': '{entanglement_type}'. Supported types are 'linear', 'circular', 'full'.")
            raise ValueError(f"Unknown entanglement_type: '{entanglement_type}'. Supported types are 'linear', 'circular', 'full'.")

        # --- 步骤 2: 初始化电路和参数计数器 ---
        qc_build = QuantumCircuit(num_qubits, description=f"Hardware-Efficient Ansatz (d={depth}, {entanglement_type})")
        param_idx = 0

        # --- 步骤 3: 循环构建每一层 ---
        for d in range(depth):
            AlgorithmBuilders._internal_logger.debug(f"[{log_prefix}] Building layer {d+1}/{depth}...")
            
            # a) 单比特旋转层 (Ry, Rz)
            if num_qubits > 0:
                AlgorithmBuilders._internal_logger.debug(f"  - Adding rotation layer for {num_qubits} qubits.")
                for q in range(num_qubits):
                    # [健 robuste性] 在每次消耗参数前进行边界检查
                    if param_idx + 1 >= len(parameters) and (param_idx < len(parameters) or len(parameters) > 0):
                        AlgorithmBuilders._internal_logger.error(f"[{log_prefix}] Parameter index out of bounds while adding Ry gate. This indicates a logic error in parameter count.")
                        raise IndexError("Parameter list exhausted prematurely.")
                    
                    qc_build.ry(q, parameters[param_idx])
                    param_idx += 1
                    
                    qc_build.rz(q, parameters[param_idx])
                    param_idx += 1
            
            # b) 纠缠层 (CNOTs)
            # 按照惯例，最后一层通常不加纠缠层，但这取决于具体的 VQA 实现。
            # 这里我们遵循一个常见的模式：只有在非最后一层才添加纠缠。
            if d < depth - 1:
                if num_qubits > 1: # 纠缠需要至少两个比特
                    AlgorithmBuilders._internal_logger.debug(f"  - Adding '{entanglement_type}' entanglement layer.")
                    if entanglement_type == 'linear':
                        for q in range(num_qubits - 1):
                            qc_build.cnot(q, q + 1)
                    elif entanglement_type == 'circular':
                        for q in range(num_qubits):
                            qc_build.cnot(q, (q + 1) % num_qubits)
                    elif entanglement_type == 'full':
                        for q1 in range(num_qubits):
                            for q2 in range(q1 + 1, num_qubits):
                                qc_build.cnot(q1, q2)
        
        AlgorithmBuilders._internal_logger.info(f"[{log_prefix}] Hardware-Efficient Ansatz with {len(qc_build.instructions)} gates constructed successfully.")
        return qc_build



# --- 在 AlgorithmBuilders 类定义的下方，注册所有宏 ---

# ========================================================================
# --- 原子门宏定义 (直接添加指令) ---
#    在双重定义系统中，这些宏对应于有优化内核的原子操作。
#    它们只负责将原始指令添加到电路中，而不执行分解。
# ========================================================================

@AlgorithmBuilders.register_macro('x', num_qubits=1, description="Pauli-X gate")
def _define_x_macro(circuit: Any, qubit_index: int, **kwargs: Any):
    """[原子宏] 直接添加 'x' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_x_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for X gate.")
    circuit._add_raw_gate("x", qubit_index, **kwargs)

@AlgorithmBuilders.register_macro('y', num_qubits=1, description="Pauli-Y gate")
def _define_y_macro(circuit: Any, qubit_index: int, **kwargs: Any):
    """[原子宏] 直接添加 'y' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_y_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for Y gate.")
    circuit._add_raw_gate("y", qubit_index, **kwargs)

@AlgorithmBuilders.register_macro('z', num_qubits=1, description="Pauli-Z gate")
def _define_z_macro(circuit: Any, qubit_index: int, **kwargs: Any):
    """[原子宏] 直接添加 'z' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_z_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for Z gate.")
    circuit._add_raw_gate("z", qubit_index, **kwargs)

@AlgorithmBuilders.register_macro('h', num_qubits=1, description="Hadamard gate")
def _define_h_macro(circuit: Any, qubit_index: int, **kwargs: Any):
    """[原子宏] 直接添加 'h' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_h_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for H gate.")
    circuit._add_raw_gate("h", qubit_index, **kwargs)

@AlgorithmBuilders.register_macro('s', num_qubits=1, description="Phase gate (S)")
def _define_s_macro(circuit: Any, qubit_index: int, **kwargs: Any):
    """[原子宏] 直接添加 's' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_s_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for S gate.")
    circuit._add_raw_gate("s", qubit_index, **kwargs)

@AlgorithmBuilders.register_macro('sx', num_qubits=1, description="Sqrt(X) gate")
def _define_sx_macro(circuit: Any, qubit_index: int, **kwargs: Any):
    """[原子宏] 直接添加 'sx' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_sx_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for SX gate.")
    circuit._add_raw_gate("sx", qubit_index, **kwargs)

@AlgorithmBuilders.register_macro('sdg', num_qubits=1, description="Phase Dagger gate (Sdg)")
def _define_sdg_macro(circuit: Any, qubit_index: int, **kwargs: Any):
    """[原子宏] 直接添加 'sdg' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_sdg_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for Sdg gate.")
    circuit._add_raw_gate("sdg", qubit_index, **kwargs)

@AlgorithmBuilders.register_macro('t_gate', num_qubits=1, description="T gate (pi/8 gate)")
def _define_t_gate_macro(circuit: Any, qubit_index: int, **kwargs: Any):
    """[原子宏] 直接添加 't_gate' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_t_gate_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for T gate.")
    circuit._add_raw_gate("t_gate", qubit_index, **kwargs)

@AlgorithmBuilders.register_macro('tdg', num_qubits=1, description="T Dagger gate (Tdg)")
def _define_tdg_macro(circuit: Any, qubit_index: int, **kwargs: Any):
    """[原子宏] 直接添加 'tdg' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_tdg_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for Tdg gate.")
    circuit._add_raw_gate("tdg", qubit_index, **kwargs)

@AlgorithmBuilders.register_macro('rx', num_qubits=1, description="Rotation around X-axis")
def _define_rx_macro(circuit: Any, qubit_index: int, theta: float, **kwargs: Any):
    """[原子宏] 直接添加 'rx' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_rx_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for RX gate.")
    if not isinstance(theta, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_rx_macro: 'theta' must be numeric, got {type(theta).__name__}.")
        raise TypeError(f"Theta must be a numeric type for RX gate.")
    circuit._add_raw_gate("rx", qubit_index, theta, **kwargs)

@AlgorithmBuilders.register_macro('ry', num_qubits=1, description="Rotation around Y-axis")
def _define_ry_macro(circuit: Any, qubit_index: int, theta: float, **kwargs: Any):
    """[原子宏] 直接添加 'ry' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_ry_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for RY gate.")
    if not isinstance(theta, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_ry_macro: 'theta' must be numeric, got {type(theta).__name__}.")
        raise TypeError(f"Theta must be a numeric type for RY gate.")
    circuit._add_raw_gate("ry", qubit_index, theta, **kwargs)

@AlgorithmBuilders.register_macro('rz', num_qubits=1, description="Rotation around Z-axis")
def _define_rz_macro(circuit: Any, qubit_index: int, phi: float, **kwargs: Any):
    """[原子宏] 直接添加 'rz' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_rz_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for RZ gate.")
    if not isinstance(phi, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_rz_macro: 'phi' must be numeric, got {type(phi).__name__}.")
        raise TypeError(f"Phi must be a numeric type for RZ gate.")
    circuit._add_raw_gate("rz", qubit_index, phi, **kwargs)

@AlgorithmBuilders.register_macro('p_gate', num_qubits=1, description="Generalized Phase gate")
def _define_p_gate_macro(circuit: Any, qubit_index: int, lambda_angle: float, **kwargs: Any):
    """[原子宏] 直接添加 'p_gate' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_p_gate_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for P gate.")
    if not isinstance(lambda_angle, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_p_gate_macro: 'lambda_angle' must be numeric, got {type(lambda_angle).__name__}.")
        raise TypeError(f"Lambda angle must be a numeric type for P gate.")
    circuit._add_raw_gate("p_gate", qubit_index, lambda_angle, **kwargs)

@AlgorithmBuilders.register_macro('u3_gate', num_qubits=1, description="Generic U3 single-qubit gate")
def _define_u3_gate_macro(circuit: Any, qubit_index: int, theta: float, phi: float, lambda_angle: float, **kwargs: Any):
    """[原子宏] 直接添加 'u3_gate' 指令。"""
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_u3_gate_macro: Invalid 'qubit_index' {qubit_index} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index {qubit_index} for U3 gate.")
    if not all(isinstance(a, (float, int)) for a in [theta, phi, lambda_angle]):
        AlgorithmBuilders._internal_logger.error(f"_define_u3_gate_macro: All angles must be numeric. Got types: ({type(theta).__name__}, {type(phi).__name__}, {type(lambda_angle).__name__}).")
        raise TypeError(f"All angles (theta, phi, lambda_angle) must be numeric for U3 gate.")
    circuit._add_raw_gate("u3_gate", qubit_index, theta, phi, lambda_angle, **kwargs)

@AlgorithmBuilders.register_macro('cnot', num_qubits=2, description="Controlled-NOT gate")
def _define_cnot_macro(circuit: Any, control: int, target: int, **kwargs: Any):
    """[原子宏] 直接添加 'cnot' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [control, target]):
        AlgorithmBuilders._internal_logger.error(f"_define_cnot_macro: Invalid control/target qubits ({control}, {target}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid control or target qubit index for CNOT gate.")
    if control == target:
        AlgorithmBuilders._internal_logger.error(f"_define_cnot_macro: Control qubit ({control}) cannot be the same as target qubit ({target}).")
        raise ValueError("Control and target qubits for CNOT gate cannot be the same.")
    circuit._add_raw_gate("cnot", control, target, **kwargs)

@AlgorithmBuilders.register_macro('cz', num_qubits=2, description="Controlled-Z gate")
def _define_cz_macro(circuit: Any, control: int, target: int, **kwargs: Any):
    """[原子宏] 直接添加 'cz' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [control, target]):
        AlgorithmBuilders._internal_logger.error(f"_define_cz_macro: Invalid control/target qubits ({control}, {target}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid control or target qubit index for CZ gate.")
    if control == target:
        AlgorithmBuilders._internal_logger.error(f"_define_cz_macro: Control qubit ({control}) cannot be the same as target qubit ({target}).")
        raise ValueError("Control and target qubits for CZ gate cannot be the same.")
    circuit._add_raw_gate("cz", control, target, **kwargs)

@AlgorithmBuilders.register_macro('cp', num_qubits=2, description="Controlled-Phase gate")
def _define_cp_macro(circuit: Any, control: int, target: int, angle: float, **kwargs: Any):
    """[原子宏] 直接添加 'cp' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [control, target]):
        AlgorithmBuilders._internal_logger.error(f"_define_cp_macro: Invalid control/target qubits ({control}, {target}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid control or target qubit index for CP gate.")
    if control == target:
        AlgorithmBuilders._internal_logger.error(f"_define_cp_macro: Control qubit ({control}) cannot be the same as target qubit ({target}).")
        raise ValueError("Control and target qubits for CP gate cannot be the same.")
    if not isinstance(angle, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_cp_macro: 'angle' must be numeric, got {type(angle).__name__}.")
        raise TypeError(f"Angle must be a numeric type for CP gate.")
    circuit._add_raw_gate("cp", control, target, angle, **kwargs)

@AlgorithmBuilders.register_macro('crx', num_qubits=2, description="Controlled-RX gate")
def _define_crx_macro(circuit: Any, control: int, target: int, theta: float, **kwargs: Any):
    """[原子宏] 直接添加 'crx' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [control, target]):
        AlgorithmBuilders._internal_logger.error(f"_define_crx_macro: Invalid control/target qubits ({control}, {target}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid control or target qubit index for CRX gate.")
    if control == target:
        AlgorithmBuilders._internal_logger.error(f"_define_crx_macro: Control qubit ({control}) cannot be the same as target qubit ({target}).")
        raise ValueError("Control and target qubits for CRX gate cannot be the same.")
    if not isinstance(theta, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_crx_macro: 'theta' must be numeric, got {type(theta).__name__}.")
        raise TypeError(f"Theta must be a numeric type for CRX gate.")
    circuit._add_raw_gate("crx", control, target, theta, **kwargs)

@AlgorithmBuilders.register_macro('cry', num_qubits=2, description="Controlled-RY gate")
def _define_cry_macro(circuit: Any, control: int, target: int, theta: float, **kwargs: Any):
    """[原子宏] 直接添加 'cry' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [control, target]):
        AlgorithmBuilders._internal_logger.error(f"_define_cry_macro: Invalid control/target qubits ({control}, {target}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid control or target qubit index for CRY gate.")
    if control == target:
        AlgorithmBuilders._internal_logger.error(f"_define_cry_macro: Control qubit ({control}) cannot be the same as target qubit ({target}).")
        raise ValueError("Control and target qubits for CRY gate cannot be the same.")
    if not isinstance(theta, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_cry_macro: 'theta' must be numeric, got {type(theta).__name__}.")
        raise TypeError(f"Theta must be a numeric type for CRY gate.")
    circuit._add_raw_gate("cry", control, target, theta, **kwargs)
        
@AlgorithmBuilders.register_macro('crz', num_qubits=2, description="Controlled-RZ gate")
def _define_crz_macro(circuit: Any, control: int, target: int, phi: float, **kwargs: Any):
    """[原子宏] 直接添加 'crz' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [control, target]):
        AlgorithmBuilders._internal_logger.error(f"_define_crz_macro: Invalid control/target qubits ({control}, {target}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid control or target qubit index for CRZ gate.")
    if control == target:
        AlgorithmBuilders._internal_logger.error(f"_define_crz_macro: Control qubit ({control}) cannot be the same as target qubit ({target}).")
        raise ValueError("Control and target qubits for CRZ gate cannot be the same.")
    if not isinstance(phi, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_crz_macro: 'phi' must be numeric, got {type(phi).__name__}.")
        raise TypeError(f"Phi must be a numeric type for CRZ gate.")
    circuit._add_raw_gate("crz", control, target, phi, **kwargs)

@AlgorithmBuilders.register_macro('controlled_u', num_qubits=2, description="Generic Controlled-U gate")
def _define_controlled_u_macro(circuit: Any, control: int, target: int, u_matrix: List[List[complex]], name: str = "CU", **kwargs: Any):
    """[原子宏] 直接添加 'controlled_u' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [control, target]):
        AlgorithmBuilders._internal_logger.error(f"_define_controlled_u_macro: Invalid control/target qubits ({control}, {target}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid control or target qubit index for Controlled-U gate.")
    if control == target:
        AlgorithmBuilders._internal_logger.error(f"_define_controlled_u_macro: Control qubit ({control}) cannot be the same as target qubit ({target}).")
        raise ValueError("Control and target qubits for Controlled-U gate cannot be the same.")
    if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
            isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
            isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2):
        AlgorithmBuilders._internal_logger.error(f"_define_controlled_u_macro: 'u_matrix' must be a 2x2 nested list of complex numbers. Got {type(u_matrix).__name__}.")
        raise ValueError("`u_matrix` for controlled_u must be a 2x2 nested list of complex numbers.")
    if not isinstance(name, str):
        AlgorithmBuilders._internal_logger.warning(f"_define_controlled_u_macro: 'name' must be a string, got {type(name).__name__}. Defaulting to 'CU'.")
        name = "CU"
    circuit._add_raw_gate("controlled_u", control, target, u_matrix, name=name, **kwargs)

@AlgorithmBuilders.register_macro('rxx', num_qubits=2, description="RXX entanglement gate")
def _define_rxx_macro(circuit: Any, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
    """[原子宏] 直接添加 'rxx' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [qubit1, qubit2]):
        AlgorithmBuilders._internal_logger.error(f"_define_rxx_macro: Invalid qubits ({qubit1}, {qubit2}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index for RXX gate.")
    if qubit1 == qubit2:
        AlgorithmBuilders._internal_logger.error(f"_define_rxx_macro: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
        raise ValueError("Qubits for RXX gate cannot be the same.")
    if not isinstance(theta, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_rxx_macro: 'theta' must be numeric, got {type(theta).__name__}.")
        raise TypeError(f"Theta must be a numeric type for RXX gate.")
    circuit._add_raw_gate("rxx", qubit1, qubit2, theta, **kwargs)

@AlgorithmBuilders.register_macro('ryy', num_qubits=2, description="RYY entanglement gate")
def _define_ryy_macro(circuit: Any, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
    """[原子宏] 直接添加 'ryy' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [qubit1, qubit2]):
        AlgorithmBuilders._internal_logger.error(f"_define_ryy_macro: Invalid qubits ({qubit1}, {qubit2}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index for RYY gate.")
    if qubit1 == qubit2:
        AlgorithmBuilders._internal_logger.error(f"_define_ryy_macro: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
        raise ValueError("Qubits for RYY gate cannot be the same.")
    if not isinstance(theta, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_ryy_macro: 'theta' must be numeric, got {type(theta).__name__}.")
        raise TypeError(f"Theta must be a numeric type for RYY gate.")
    circuit._add_raw_gate("ryy", qubit1, qubit2, theta, **kwargs)
        
@AlgorithmBuilders.register_macro('rzz', num_qubits=2, description="RZZ entanglement gate")
def _define_rzz_macro(circuit: Any, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
    """[原子宏] 直接添加 'rzz' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [qubit1, qubit2]):
        AlgorithmBuilders._internal_logger.error(f"_define_rzz_macro: Invalid qubits ({qubit1}, {qubit2}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index for RZZ gate.")
    if qubit1 == qubit2:
        AlgorithmBuilders._internal_logger.error(f"_define_rzz_macro: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
        raise ValueError("Qubits for RZZ gate cannot be the same.")
    if not isinstance(theta, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_rzz_macro: 'theta' must be numeric, got {type(theta).__name__}.")
        raise TypeError(f"Theta must be a numeric type for RZZ gate.")
    circuit._add_raw_gate("rzz", qubit1, qubit2, theta, **kwargs)

@AlgorithmBuilders.register_macro('mcz', num_qubits=0, description="Multi-Controlled-Z gate (variable controls)")
def _define_mcz_macro(circuit: Any, controls: List[int], target: int, **kwargs: Any):
    """[原子宏] 直接添加 'mcz' 指令。"""
    if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in controls):
        AlgorithmBuilders._internal_logger.error(f"_define_mcz_macro: 'controls' must be a list of valid qubit indices, got {controls}.")
        raise ValueError(f"Invalid 'controls' list for MCZ gate.")
    if not isinstance(target, int) or not (0 <= target < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_mcz_macro: 'target' must be a valid qubit index, got {target}.")
        raise ValueError(f"Invalid target qubit index for MCZ gate.")
    
    all_qubits = controls + [target]
    if len(set(all_qubits)) != len(all_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_mcz_macro: All qubits (controls and target) must be distinct. Got controls={controls}, target={target}.")
        raise ValueError("All qubits (controls and target) for MCZ gate must be distinct.")
    circuit._add_raw_gate("mcz", controls, target, **kwargs)

@AlgorithmBuilders.register_macro('mcp', num_qubits=0, description="Multi-Controlled-Phase gate (variable controls)")
def _define_mcp_macro(circuit: Any, controls: List[int], target: int, angle: float, **kwargs: Any):
    """[原子宏] 直接添加 'mcp' 指令。"""
    if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in controls):
        AlgorithmBuilders._internal_logger.error(f"_define_mcp_macro: 'controls' must be a list of valid qubit indices, got {controls}.")
        raise ValueError(f"Invalid 'controls' list for MCP gate.")
    if not isinstance(target, int) or not (0 <= target < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_mcp_macro: 'target' must be a valid qubit index, got {target}.")
        raise ValueError(f"Invalid target qubit index for MCP gate.")
    if not isinstance(angle, (float, int)):
        AlgorithmBuilders._internal_logger.error(f"_define_mcp_macro: 'angle' must be numeric, got {type(angle).__name__}.")
        raise TypeError(f"Angle must be a numeric type for MCP gate.")
    
    all_qubits = controls + [target]
    if len(set(all_qubits)) != len(all_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_mcp_macro: All qubits (controls and target) must be distinct. Got controls={controls}, target={target}.")
        raise ValueError("All qubits (controls and target) for MCP gate must be distinct.")
    circuit._add_raw_gate("mcp", controls, target, angle, **kwargs)

@AlgorithmBuilders.register_macro('mcu', num_qubits=0, description="Generic Multi-Controlled-U gate (variable controls)")
def _define_mcu_macro(circuit: Any, controls: List[int], target: int, u_matrix: List[List[complex]], name: str = "MCU", **kwargs: Any):
    """[原子宏] 直接添加 'mcu' 指令。"""
    if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in controls):
        AlgorithmBuilders._internal_logger.error(f"_define_mcu_macro: 'controls' must be a list of valid qubit indices, got {controls}.")
        raise ValueError(f"Invalid 'controls' list for MCU gate.")
    if not isinstance(target, int) or not (0 <= target < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_mcu_macro: 'target' must be a valid qubit index, got {target}.")
        raise ValueError(f"Invalid target qubit index for MCU gate.")
    if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
            isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
            isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2):
        AlgorithmBuilders._internal_logger.error(f"_define_mcu_macro: 'u_matrix' must be a 2x2 nested list of complex numbers. Got {type(u_matrix).__name__}.")
        raise ValueError("`u_matrix` for mcu must be a 2x2 nested list of complex numbers.")
    if not isinstance(name, str):
        AlgorithmBuilders._internal_logger.warning(f"_define_mcu_macro: 'name' must be a string, got {type(name).__name__}. Defaulting to 'MCU'.")
        name = "MCU"
    
    all_qubits = controls + [target]
    if len(set(all_qubits)) != len(all_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_mcu_macro: All qubits (controls and target) must be distinct. Got controls={controls}, target={target}.")
        raise ValueError("All qubits (controls and target) for MCU gate must be distinct.")
    circuit._add_raw_gate("mcu", controls, target, u_matrix, name=name, **kwargs)

# ========================================================================
# --- 复合门宏定义 (通过调用其他宏或基础门) ---
#   在此双重定义系统中，这些函数分为两类：
#   1. 原子宏: 对于有优化内核的门，只添加原始指令 (e.g., toffoli, mcx)。
#   2. 复合宏: 对于没有内核的门，执行分解 (e.g., swap, fredkin)。
# ========================================================================

@AlgorithmBuilders.register_macro('swap', num_qubits=2, description="SWAP gate (decomposed into CNOTs)")
def _define_swap_macro(circuit: Any, qubit1: int, qubit2: int, **kwargs: Any):
    """
    [复合宏] 将 SWAP 门分解为三个 CNOT 门。
    SWAP 门没有专门的优化内核，因此必须进行分解。
    """
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [qubit1, qubit2]):
        AlgorithmBuilders._internal_logger.error(f"_define_swap_macro: Invalid qubits ({qubit1}, {qubit2}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index for SWAP gate.")
    if qubit1 == qubit2:
        AlgorithmBuilders._internal_logger.error(f"_define_swap_macro: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
        raise ValueError("Qubits for SWAP gate cannot be the same.")
    
    # 调用 cnot 宏三次以实现 SWAP
    circuit.cnot(qubit1, qubit2, **kwargs)
    circuit.cnot(qubit2, qubit1, **kwargs)
    circuit.cnot(qubit1, qubit2, **kwargs)

@AlgorithmBuilders.register_macro('toffoli', num_qubits=3, description="Toffoli (CCX) gate")
def _define_toffoli_macro(circuit: Any, control_1: int, control_2: int, target: int, **kwargs: Any):
    """
    [原子宏] 直接添加 'toffoli' 指令。
    此宏不再执行分解，因为 Toffoli 门有专门的优化内核。
    """
    qubits = [control_1, control_2, target]
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_toffoli_macro: Invalid qubit indices {qubits} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index for Toffoli gate.")
    if len(set(qubits)) != 3:
        AlgorithmBuilders._internal_logger.error(f"_define_toffoli_macro: All qubits for Toffoli must be distinct. Got {qubits}.")
        raise ValueError("All qubits for Toffoli gate must be distinct.")

    # 直接添加原始指令，而不是分解
    circuit._add_raw_gate("toffoli", control_1, control_2, target, **kwargs)

@AlgorithmBuilders.register_macro('mcx', num_qubits=0, description="Multi-Controlled-X gate (variable controls)")
def _define_mcx_macro(circuit: Any, controls: List[int], target: int, **kwargs: Any):
    """
    [原子宏] 直接添加 'mcx' 指令。
    MCX 门有专门的优化内核，能够处理任意数量的控制比特。
    """
    if not isinstance(controls, list) or not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in controls):
        AlgorithmBuilders._internal_logger.error(f"_define_mcx_macro: 'controls' must be a list of valid qubit indices, got {controls}.")
        raise ValueError(f"Invalid 'controls' list for MCX gate.")
    if not isinstance(target, int) or not (0 <= target < circuit.num_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_mcx_macro: 'target' must be a valid qubit index, got {target}.")
        raise ValueError(f"Invalid target qubit index for MCX gate.")
    
    all_qubits = controls + [target]
    if len(set(all_qubits)) != len(all_qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_mcx_macro: All qubits (controls and target) must be distinct. Got controls={controls}, target={target}.")
        raise ValueError("All qubits (controls and target) for MCX gate must be distinct.")
    
    # 直接添加原始指令，而不是分解
    circuit._add_raw_gate("mcx", controls, target, **kwargs)

@AlgorithmBuilders.register_macro('fredkin', num_qubits=3, description="Fredkin (CSWAP) gate decomposition")
def _define_fredkin_macro(circuit: Any, control: int, target_1: int, target_2: int, **kwargs: Any):
    """
    [复合宏] 将 Fredkin (CSWAP) 门分解为 CNOT 和 Toffoli 门。
    Fredkin 门没有专门的优化内核，因此必须进行分解。
    """
    qubits = [control, target_1, target_2]
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in qubits):
        AlgorithmBuilders._internal_logger.error(f"_define_fredkin_macro: Invalid qubit indices {qubits} for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index for Fredkin gate.")
    if len(set(qubits)) != 3:
        AlgorithmBuilders._internal_logger.error(f"_define_fredkin_macro: All qubits for Fredkin must be distinct. Got {qubits}.")
        raise ValueError("All qubits for Fredkin gate must be distinct.")
    
    # 标准的 Fredkin 分解
    circuit.cnot(target_2, target_1, **kwargs)
    circuit.toffoli(control, target_1, target_2, **kwargs) # 调用 toffoli (原子宏)
    circuit.cnot(target_2, target_1, **kwargs)

@AlgorithmBuilders.register_macro('iswap', num_qubits=2, description="iSWAP gate decomposition")
def _define_iswap_macro(circuit: Any, qubit1: int, qubit2: int, **kwargs: Any):
    """
    [复合宏] 将 iSWAP 门分解为 RXX 和 RYY 门。
    iSWAP 门没有专门的优化内核，因此必须进行分解。
    分解序列为： iSWAP = RXX(-π/2) * RYY(-π/2)
    """
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [qubit1, qubit2]):
        AlgorithmBuilders._internal_logger.error(f"_define_iswap_macro: Invalid qubits ({qubit1}, {qubit2}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index for iSWAP gate.")
    if qubit1 == qubit2:
        AlgorithmBuilders._internal_logger.error(f"_define_iswap_macro: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
        raise ValueError("Qubits for iSWAP gate cannot be the same.")

    angle = -math.pi / 2.0
    
    # 调用 rxx 和 ryy (原子宏)
    circuit.rxx(qubit1, qubit2, angle, **kwargs)
    circuit.ryy(qubit1, qubit2, angle, **kwargs)

@AlgorithmBuilders.register_macro('ecr', num_qubits=2, description="ECR (Echoed Cross-Resonance) gate decomposition")
def _define_ecr_macro(circuit: Any, qubit1: int, qubit2: int, **kwargs: Any):
    """
    [复合宏] 将 ECR 门分解为 RZ, RX 和 CNOT 门。
    ECR 门没有专门的优化内核，因此必须进行分解。
    """
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [qubit1, qubit2]):
        AlgorithmBuilders._internal_logger.error(f"_define_ecr_macro: Invalid qubits ({qubit1}, {qubit2}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index for ECR gate.")
    if qubit1 == qubit2:
        AlgorithmBuilders._internal_logger.error(f"_define_ecr_macro: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
        raise ValueError("Qubits for ECR gate cannot be the same.")
    
    PI_HALF = math.pi / 2
    # 调用 rz, rx, cnot (均为原子宏)
    circuit.rz(qubit1, PI_HALF, **kwargs)
    circuit.rx(qubit1, PI_HALF, **kwargs)
    circuit.cnot(qubit1, qubit2, **kwargs)
    circuit.rx(qubit2, -PI_HALF, **kwargs)
    circuit.rz(qubit1, -PI_HALF, **kwargs)

@AlgorithmBuilders.register_macro('ecrdg', num_qubits=2, description="ECR-dagger (ECR†) gate decomposition")
def _define_ecrdg_macro(circuit: Any, qubit1: int, qubit2: int, **kwargs: Any):
    """
    [复合宏] 将 ECR-dagger 门分解为 RZ, RX 和 CNOT 门。
    ECRdg 门没有专门的优化内核，因此必须进行分解。
    """
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in [qubit1, qubit2]):
        AlgorithmBuilders._internal_logger.error(f"_define_ecrdg_macro: Invalid qubits ({qubit1}, {qubit2}) for circuit with {circuit.num_qubits} qubits.")
        raise ValueError(f"Invalid qubit index for ECRdg gate.")
    if qubit1 == qubit2:
        AlgorithmBuilders._internal_logger.error(f"_define_ecrdg_macro: Qubit1 ({qubit1}) cannot be the same as Qubit2 ({qubit2}).")
        raise ValueError("Qubits for ECRdg gate cannot be the same.")
            
    PI_HALF = math.pi / 2
    # 分解序列是 ECR 分解的逆序，且旋转角度取反
    circuit.rz(qubit1, PI_HALF, **kwargs)
    circuit.rx(qubit2, PI_HALF, **kwargs)
    circuit.cnot(qubit1, qubit2, **kwargs)
    circuit.rx(qubit1, -PI_HALF, **kwargs)
    circuit.rz(qubit1, -PI_HALF, **kwargs)

@AlgorithmBuilders.register_macro('barrier', num_qubits=0, description="Barrier (logical separator, no quantum operation)")
def _define_barrier_macro(circuit: Any, *qubits: int, **kwargs: Any):
    """[原子宏] 直接添加 'barrier' 指令。"""
    if not all(isinstance(q, int) and 0 <= q < circuit.num_qubits for q in qubits):
        AlgorithmBuilders._internal_logger.warning(f"_define_barrier_macro: Some qubits {qubits} for barrier are out of range for circuit with {circuit.num_qubits} qubits. Storing as is.")
    circuit._add_raw_gate("barrier", *qubits, **kwargs)


# ========================================================================
# --- 9. [核心] 公共 API (Public API) ---
# ========================================================================

# [健壮性改进] 添加日志器
_public_api_logger = logging.getLogger(f"{__name__}.PublicAPI")

def print_progress_bar(iteration: int, total: int, prefix: str = '进度', suffix: str = '完成', length: int = 50, fill: str = '█'):
    """
    [健壮性改进版] 在控制台打印一个可动态更新的文本进度条。

    这是一个通用的工具函数，可被任何需要显示长时间任务进度的应用直接调用。
    它通过使用回车符 `\r` 将光标移动到行首来覆盖上一行输出，从而实现
    在同一行内动态更新进度条的效果。为了确保立即显示，每次打印后都会
    刷新标准输出缓冲区。

    Args:
        iteration (int):
            当前已完成的迭代次数。必须是非负整数。
        total (int):
            任务的总迭代次数。必须是非负整数。
        prefix (str, optional):
            显示在进度条前面的文本。默认为 '进度'。
        suffix (str, optional):
            显示在进度条后面的文本。默认为 '完成'。
        length (int, optional):
            进度条自身的字符长度（不包括前缀和后缀）。默认为 50。
        fill (str, optional):
            用于填充进度条已完成部分的字符。默认为 '█'。

    Raises:
        TypeError: 如果 `iteration` 或 `total` 不是整数，或 `prefix`, `suffix`, `fill` 不是字符串，`length` 不是整数。
        ValueError: 如果 `iteration` 或 `total` 为负数，或 `length` 为负数。
    """
    # --- 1. 参数类型和值检查 ---
    if not isinstance(iteration, int):
        _public_api_logger.error(f"print_progress_bar: 'iteration' must be an integer, got {type(iteration).__name__}.")
        raise TypeError("'iteration' must be an integer.")
    if not isinstance(total, int):
        _public_api_logger.error(f"print_progress_bar: 'total' must be an integer, got {type(total).__name__}.")
        raise TypeError("'total' must be an integer.")
    if iteration < 0:
        _public_api_logger.error(f"print_progress_bar: 'iteration' cannot be negative, got {iteration}.")
        raise ValueError("'iteration' cannot be negative.")
    if total < 0:
        _public_api_logger.error(f"print_progress_bar: 'total' cannot be negative, got {total}.")
        raise ValueError("'total' cannot be negative.")
    if not isinstance(prefix, str):
        _public_api_logger.error(f"print_progress_bar: 'prefix' must be a string, got {type(prefix).__name__}.")
        raise TypeError("'prefix' must be a string.")
    if not isinstance(suffix, str):
        _public_api_logger.error(f"print_progress_bar: 'suffix' must be a string, got {type(suffix).__name__}.")
        raise TypeError("'suffix' must be a string.")
    if not isinstance(length, int) or length < 0:
        _public_api_logger.error(f"print_progress_bar: 'length' must be a non-negative integer, got {length}.")
        raise ValueError("'length' must be a non-negative integer.")
    if not isinstance(fill, str) or len(fill) != 1:
        _public_api_logger.error(f"print_progress_bar: 'fill' must be a single character string, got '{fill}'.")
        raise ValueError("'fill' must be a single character string.")

    # --- 2. 处理 total 为零的边缘情况 ---
    if total == 0:
        percent = "0.0"
        filled_length = 0
    else:
        current_iteration = min(iteration, total) # 确保 iteration 不超过 total
        percent = ("{0:.1f}").format(100 * (current_iteration / total))
        filled_length = int(length * current_iteration // total)
        
    # --- 3. 构建进度条字符串 ---
    bar = fill * filled_length + '-' * (length - filled_length)
    
    # --- 4. 打印完整的进度条行并刷新 ---
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

def _get_backend_instance() -> Union['PurePythonBackend', 'CuPyBackendWrapper']:
    """
    [内部辅助函数] 根据全局配置安全地创建并返回一个后端实例。
    此函数是轻量级的，没有副作用，旨在打破循环依赖。
    """
    backend_choice = _core_config.get("BACKEND_CHOICE", "auto").lower()
    
    global cp
    if cp is None and backend_choice in ['auto', 'cupy']:
        try:
            import cupy as cp_actual
            cp = cp_actual
        except ImportError:
            cp = None

    if backend_choice == 'auto':
        return CuPyBackendWrapper(cp) if cp is not None else PurePythonBackend()
    elif backend_choice == 'cupy':
        if cp is None:
            raise ImportError("CuPy backend was requested, but the 'cupy' library is not installed or failed to import.")
        return CuPyBackendWrapper(cp)
    elif backend_choice == 'pure_python':
        return PurePythonBackend()
    else:
        raise ValueError(f"Invalid BACKEND_CHOICE in configuration: '{backend_choice}'.")
def create_quantum_state(num_qubits: int) -> 'QuantumState':
    """
    [v1.5.4 修复版] 创建一个处于 |0...0⟩ 态的初始惰性量子态。
    此版本使用 _get_backend_instance 来避免循环依赖。
    """
    _public_api_logger.debug(f"API: Attempting to create a new quantum state with {num_qubits} qubits.")
    
    if not isinstance(num_qubits, int) or num_qubits < 0:
        _public_api_logger.error(f"API: Invalid 'num_qubits' {num_qubits}. Must be a non-negative integer.")
        raise ValueError("num_qubits must be a non-negative integer.")

    try:
        # 直接实例化 QuantumState，它将在 __post_init__ 中调用 _get_backend_instance
        state = QuantumState(num_qubits=num_qubits)
        
        backend_name = type(state._backend).__name__
        _public_api_logger.info(f"API: Successfully created a {num_qubits}-qubit lazy quantum state container using the {backend_name} backend.")
        
        return state
        
    except (ValueError, ImportError, MemoryError) as e:
        _public_api_logger.critical(f"API: Failed to create quantum state with {num_qubits} qubits. Reason: {e}", exc_info=True)
        raise
    except Exception as e:
        _public_api_logger.critical(f"API: An unexpected error occurred during QuantumState creation for {num_qubits} qubits: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during QuantumState creation: {e}") from e

def get_effective_unitary(circuit: 'QuantumCircuit', backend_choice: str = 'auto') -> Any:
    """
    [公共API] [v1.6 解耦增强版] 计算并返回给定 QuantumCircuit 对应的全局有效酉矩阵。

    此函数是计算任意量子线路等效酉矩阵的官方、安全入口。它采用了一种
    高效且与 `quantum_core` 执行引擎逻辑完全一致的方法，即“矩阵累积模式”。

    核心特性:
    - 模式感知执行: 内部创建一个临时的 `_StateVectorEntity`，并将其状态初始化
      为单位矩阵。当线路在该实体上运行时，实体会自动将门操作解释为矩阵乘法，
      从而高效地累积出最终的全局酉矩阵。
    - 后端选择: 新增的 `backend_choice` 参数允许调用者强制在特定后端
      （如 'pure_python' 或 'cupy'）上进行计算。这对于需要确定性数值行为
      （如在优化器中进行验证）或强制使用特定硬件的场景至关重要。
    - 严格的输入验证: 对 `circuit` 和 `backend_choice` 参数进行全面的类型
      和值检查。
    - 清晰的错误处理: 捕获在后端初始化、内存分配或矩阵计算过程中可能发生的
      所有错误，并将其包装为带有清晰上下文的 `RuntimeError`。

    Args:
        circuit (QuantumCircuit):
            要计算其等效酉矩阵的 `QuantumCircuit` 对象。
        backend_choice (str, optional):
            指定用于计算的后端。可选值为 'auto', 'pure_python', 'cupy'。
            - 'auto': 自动选择最佳可用后端（优先使用CuPy）。
            - 'pure_python': 强制使用纯Python后端，保证数值行为的确定性。
            - 'cupy': 强制使用CuPy后端，如果CuPy不可用则会报错。
            默认为 'auto'。

    Returns:
        Any:
            一个代表电路全局有效酉矩阵的后端特定对象（Python嵌套列表或CuPy数组）。

    Raises:
        TypeError: 如果 'circuit' 不是 QuantumCircuit 实例。
        ValueError: 如果 'backend_choice' 无效。
        ImportError: 如果请求了 'cupy' 后端但CuPy库不可用。
        RuntimeError: 如果在计算过程中发生任何内部错误（如内存不足）。
    """
    # 使用公共API专用的日志记录器
    _public_api_logger.debug(f"API: get_effective_unitary called for circuit '{circuit.description or 'unnamed'}' (N={circuit.num_qubits}), backend_choice='{backend_choice}'.")

    # --- 步骤 1: 严格的输入验证 ---
    if not isinstance(circuit, QuantumCircuit):
        _public_api_logger.error(f"API: Input to get_effective_unitary must be a QuantumCircuit instance, got {type(circuit).__name__}.")
        raise TypeError("Input to get_effective_unitary must be a QuantumCircuit instance.")
    
    # --- 步骤 2: 根据 backend_choice 参数获取指定的后端实例 ---
    try:
        if not isinstance(backend_choice, str) or backend_choice.lower() not in ['auto', 'pure_python', 'cupy']:
            raise ValueError(f"Invalid backend_choice '{backend_choice}'. Must be 'auto', 'pure_python', or 'cupy'.")
        
        backend_choice_lower = backend_choice.lower()
        
        backend: Union['PurePythonBackend', 'CuPyBackendWrapper']
        if backend_choice_lower == 'auto':
            backend = _get_backend_instance()
        elif backend_choice_lower == 'pure_python':
            backend = PurePythonBackend()
        elif backend_choice_lower == 'cupy':
            # 尝试导入或确认 CuPy 是否存在
            if cp is None:
                raise ImportError("CuPy backend was explicitly requested, but the 'cupy' library is not installed or failed to import.")
            backend = CuPyBackendWrapper(cp)
        
    except (ValueError, ImportError) as e:
        _public_api_logger.critical(f"API: Failed to initialize backend '{backend_choice}' for get_effective_unitary: {e}", exc_info=True)
        # 直接重新抛出，因为这是由用户输入引起的配置错误
        raise e
    except Exception as e:
        _public_api_logger.critical(f"API: An unexpected error occurred while initializing backend '{backend_choice}': {e}", exc_info=True)
        raise RuntimeError(f"Could not determine the computation backend for the {circuit.num_qubits}-qubit circuit.") from e

    # --- 步骤 3: 处理 0-qubit 边缘情况 ---
    num_qubits = circuit.num_qubits
    if num_qubits == 0:
        _public_api_logger.debug("API: Circuit has 0 qubits, returning 1x1 identity matrix.")
        # 返回一个 1x1 的单位矩阵，其类型与所选后端匹配
        return backend.eye(1, dtype=complex)

    _public_api_logger.info(f"API: Calculating effective unitary for circuit '{circuit.description or 'unnamed'}' (N={num_qubits}) on backend '{type(backend).__name__}'...")

    # --- 步骤 4: 执行“矩阵累积模式”计算 ---
    try:
        # a) 创建一个临时的、非惰性的计算实体
        entity_for_calc = _StateVectorEntity(num_qubits, backend)
        
        # b) [核心技巧] 将其内部状态替换为单位矩阵，进入“矩阵累积模式”
        dim = 1 << num_qubits
        entity_for_calc._state_vector = backend.eye(dim, dtype=complex)
        
        # c) 在该实体上运行电路。实体内部的 run_circuit_on_entity 方法是模式感知的，
        #    它会检测到内部状态是二维矩阵，并自动将门操作解释为矩阵乘法。
        entity_for_calc.run_circuit_on_entity(circuit)
        
        # d) 提取最终累积的酉矩阵
        effective_unitary = entity_for_calc._state_vector
        
        _public_api_logger.info("API: Effective unitary calculation completed successfully.")
        return effective_unitary

    except Exception as e:
        # --- 步骤 5: 捕获所有可能的底层计算错误 ---
        # 这包括内存不足、数值错误或来自后端的任何其他异常。
        _public_api_logger.critical(f"API: An error occurred while calculating effective unitary for circuit '{circuit.description or 'unnamed'}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to compute effective unitary for circuit '{circuit.description or 'unnamed'}'.") from e
# [健壮性改进] 确保在 get_effective_unitary 定义之后，将其赋给全局占位符
_get_effective_unitary_placeholder = get_effective_unitary

def run_circuit_on_state(state: 'QuantumState',
                         circuit: 'QuantumCircuit',
                         noise_model: Optional['NoiseModel'] = None,
                         topology: Optional[Dict[int, List[int]]] = None) -> 'QuantumState':
    """
    [健壮性改进版] 在一个量子态上执行一个量子线路，返回演化后的新量子态。
    
    此函数现在是一个简洁的包装器，它遵循函数式编程的最佳实践：
    1.  创建输入`state`的一个完整深拷贝，以保证原始状态的不可变性。
    2.  将所有复杂的执行逻辑（包括模式切换、惰性求值、断点处理和噪声应用）
        完全委托给 `QuantumState` 对象自身的 `run_circuit` 方法。
    3.  返回经过演化后的新状态对象。

    Args:
        state (QuantumState):
            执行线路前的初始量子态。
        circuit (QuantumCircuit):
            包含一系列待执行量子操作的`QuantumCircuit`对象。
        noise_model (Optional[NoiseModel]):
            一个可选的噪声模型实例。如果提供，模拟将包含噪声效应。
        topology (Optional[Dict[int, List[int]]]):
            一个可选的硬件拓扑图。如果提供，它将被传递给内部的
            `nexus_optimizer` 以执行拓扑感知的编译（例如，SWAP门插入）。

    Returns:
        QuantumState:
            一个表示演化后状态的**新**`QuantumState`实例。
            
    Raises:
        TypeError: 如果输入参数类型不正确。
        ValueError: 如果线路比特数不匹配。
        RuntimeError: 如果线路执行失败。
    """
    _public_api_logger.debug(f"API: run_circuit_on_state called for N={state.num_qubits} with circuit '{circuit.description or 'unnamed'}'")

    # --- 1. 输入验证 ---
    if not isinstance(state, QuantumState):
        _public_api_logger.error(f"API: Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    if not isinstance(circuit, QuantumCircuit):
        _public_api_logger.error(f"API: Input 'circuit' must be a QuantumCircuit instance, but got {type(circuit).__name__}.")
        raise TypeError(f"Input 'circuit' must be a QuantumCircuit instance, but got {type(circuit).__name__}.")
    if circuit.num_qubits > state.num_qubits:
        _public_api_logger.error(f"API: State ({state.num_qubits} qubits) cannot execute a circuit for more qubits ({circuit.num_qubits}).")
        raise ValueError(f"State ({state.num_qubits} qubits) cannot execute a circuit designed for more qubits ({circuit.num_qubits} qubits).")
    if noise_model is not None and not isinstance(noise_model, NoiseModel):
        _public_api_logger.error(f"API: 'noise_model' must be a NoiseModel instance or None, got {type(noise_model).__name__}.")
        raise TypeError(f"'noise_model' must be a NoiseModel instance or None.")
    if topology is not None and not isinstance(topology, dict):
        _public_api_logger.error(f"API: 'topology' must be a dictionary or None, got {type(topology).__name__}.")
        raise TypeError(f"'topology' must be a dictionary or None.")

    # --- 2. 深拷贝以保证不可变性 ---
    try:
        evolved_state = copy.deepcopy(state)
    except Exception as e:
        _public_api_logger.critical(f"API: Failed to deepcopy QuantumState for circuit execution: {e}", exc_info=True)
        raise RuntimeError(f"Failed to deepcopy QuantumState before running circuit: {e}") from e
    
    # --- 3. 将执行完全委托给 QuantumState 对象自身的方法 ---
    try:
        evolved_state.run_circuit(circuit, noise_model, topology=topology)
    except Exception as e:
        _public_api_logger.critical(f"API: Circuit execution failed for circuit '{circuit.description or 'unnamed'}': {e}", exc_info=True)
        raise RuntimeError(f"Circuit execution failed: {e}") from e
    
    _public_api_logger.info(f"API: Circuit execution completed. Final state mode: '{evolved_state._simulation_mode}'.")
    
    # --- 4. 返回演化后的新状态 ---
    return evolved_state
def get_state_data(state: 'QuantumState', format: str = 'python_list') -> Union[List[complex], List[List[complex]]]:
    """
    [公共API] [健壮性改进版] 安全地从一个 QuantumState 对象中提取底层的状态数据。

    此函数是访问惰性量子态计算结果的官方、安全入口。它会根据需要触发
    惰性求值的量子态的展开计算，然后返回其内部的态矢量或密度矩阵。

    为了确保上层应用的健壮性和后端无关性，默认情况下，它会返回与后端无关的
    Python原生列表格式。

    核心特性:
    - 模式感知 (Mode-Aware): 自动检测当前状态是处于 'statevector' 还是
      'density_matrix' 模式，并返回相应的数据结构。
    - 惰性求值触发: 如果状态处于惰性模式（即缓存无效），此函数会首先调用
      内部的 `_expand_to_statevector` 方法来执行所有累积的电路指令。
    - 后端无关的输出: 默认将后端特定的数据结构（如 CuPy 数组）安全地转换
      为标准的 Python 列表或嵌套列表。
    - 严格的输入验证: 对所有输入参数的类型和值进行检查。
    - 清晰的错误处理: 捕获在状态展开或数据提取过程中可能发生的任何错误，
      并将其包装为带有清晰上下文的 `RuntimeError`。

    Args:
        state (QuantumState):
            要提取数据的量子态对象。
        format (str, optional):
            返回数据的格式。当前仅支持 'python_list'。默认为 'python_list'。

    Returns:
        Union[List[complex], List[List[complex]]]:
            如果处于 'statevector' 模式，返回一个表示态矢量的一维列表。
            如果处于 'density_matrix' 模式，返回一个表示密度矩阵的嵌套列表。

    Raises:
        TypeError: 如果输入 'state' 不是 QuantumState 实例。
        ValueError: 如果请求了不支持的格式，或内部状态模式未知。
        RuntimeError: 如果状态展开或数据提取失败。
    """
    # 使用公共API专用的日志记录器
    _public_api_logger.debug(f"API: get_state_data called for N={state.num_qubits}. Mode: '{state._simulation_mode}', Format: '{format}'.")

    # --- 步骤 1: 严格的输入验证 ---
    if not isinstance(state, QuantumState):
        _public_api_logger.error(f"API: Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
        raise TypeError(f"Input 'state' must be a QuantumState instance.")

    if not isinstance(format, str) or format != 'python_list':
        _public_api_logger.error(f"API: Unsupported format '{format}' requested. Currently only 'python_list' is supported.")
        raise ValueError(f"Unsupported format '{format}'.")

    try:
        # --- 步骤 2: 根据模式触发展开并获取后端特定的数据 ---
        data_backend_specific: Any
        
        if state._simulation_mode == 'statevector':
            # a) 触发展开计算（如果缓存无效）
            state._expand_to_statevector()
            
            # b) 验证缓存实体是否存在
            if state._cached_statevector_entity is None or state._cached_statevector_entity._state_vector is None:
                _public_api_logger.critical(f"API: State expansion failed or resulted in a null entity for a {state.num_qubits}-qubit state.")
                raise RuntimeError("State expansion failed or resulted in a null entity.")
            
            # c) 获取后端特定的态矢量数据
            data_backend_specific = state._cached_statevector_entity._state_vector
        
        elif state._simulation_mode == 'density_matrix':
            # a) 验证密度矩阵是否存在
            if state._density_matrix is None:
                _public_api_logger.critical(f"API: Density matrix is None in 'density_matrix' mode for a {state.num_qubits}-qubit state.")
                raise RuntimeError("Density matrix is None in 'density_matrix' mode.")
            
            # b) 获取后端特定的密度矩阵数据
            data_backend_specific = state._density_matrix
        
        else:
            # d) 处理未知的内部状态模式
            _public_api_logger.critical(f"API: Unknown simulation mode '{state._simulation_mode}' encountered.")
            raise ValueError(f"Unknown simulation mode '{state._simulation_mode}'.")

        # --- 步骤 3: 将后端特定的数据安全地转换为 Python 原生列表 ---
        
        # 检查对象是否具有 .tolist() 方法（这是 CuPy/NumPy 数组的标准方法）
        if hasattr(data_backend_specific, 'tolist'):
            _public_api_logger.debug("API: Converting backend array to Python list using .tolist().")
            return data_backend_specific.tolist()
        
        # 如果它本身就是 Python 列表，则返回其深拷贝以防止外部修改
        elif isinstance(data_backend_specific, list):
            _public_api_logger.debug("API: Data is already a Python list, returning a deep copy.")
            return copy.deepcopy(data_backend_specific)
        
        # 如果以上方法都失败，说明后端返回了一个未知的数据类型
        else:
            _public_api_logger.error(f"API: Cannot convert backend data of type {type(data_backend_specific).__name__} to a Python list.")
            raise TypeError(f"Cannot convert backend data of type {type(data_backend_specific).__name__} to a list.")

    except Exception as e:
        # --- 步骤 4: 捕获所有可能的异常并重新包装 ---
        # 捕获在 _expand_to_statevector, tolist(), deepcopy() 等过程中可能发生的所有错误
        
        # 避免重复包装 RuntimeError
        if isinstance(e, RuntimeError):
            _public_api_logger.critical(f"API: A runtime error occurred during get_state_data: {e}", exc_info=True)
            raise e
        else:
            _public_api_logger.critical(f"API: An unexpected error occurred while getting state data: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get state data due to an internal error: {e}") from e
def calculate_fidelity_sv(vec1: Any, vec2: Any, backend: Any) -> float:
    """
    [健壮性改进版] 计算两个态矢量之间的保真度。
    Fidelity = |⟨ψ1|ψ2⟩|^2

    Args:
        vec1 (Any): 第一个态矢量（Python列表或CuPy数组）。
        vec2 (Any): 第二个态矢量（Python列表或CuPy数组）。
        backend (Any): 计算后端实例 (PurePythonBackend or CuPyBackendWrapper)。

    Returns:
        float: 计算出的保真度值，范围在 [0.0, 1.0] 之间。

    Raises:
        TypeError: 如果输入参数类型不正确。
        ValueError: 如果态矢量为空或维度不匹配。
        RuntimeError: 如果底层计算失败。
    """
    # [健壮性改进] 输入验证
    if backend is None or not isinstance(backend, (PurePythonBackend, CuPyBackendWrapper)):
        _public_api_logger.error(f"API: 'backend' must be a PurePythonBackend or CuPyBackendWrapper instance, got {type(backend).__name__}.")
        raise TypeError("A valid backend must be provided to calculate fidelity.")
    
    # 将输入转换为后端兼容的类型
    vec1_backend = backend._ensure_cupy_array(vec1) if isinstance(backend, CuPyBackendWrapper) else vec1
    vec2_backend = backend._ensure_cupy_array(vec2) if isinstance(backend, CuPyBackendWrapper) else vec2

    if not isinstance(vec1_backend, list) and not hasattr(vec1_backend, 'shape'):
        _public_api_logger.error(f"API: 'vec1' must be a list or CuPy array, got {type(vec1).__name__}.")
        raise TypeError("'vec1' must be a list or CuPy array.")
    if not isinstance(vec2_backend, list) and not hasattr(vec2_backend, 'shape'):
        _public_api_logger.error(f"API: 'vec2' must be a list or CuPy array, got {type(vec2).__name__}.")
        raise TypeError("'vec2' must be a list or CuPy array.")
    
    len1 = len(vec1_backend) if isinstance(vec1_backend, list) else vec1_backend.size
    len2 = len(vec2_backend) if isinstance(vec2_backend, list) else vec2_backend.size

    if len1 == 0 or len2 == 0:
        _public_api_logger.warning("API: One or both state vectors are empty. Fidelity is 0.0.")
        return 0.0
    if len1 != len2:
        _public_api_logger.error(f"API: State vector dimensions mismatch: {len1} vs {len2}.")
        raise ValueError("State vector dimensions mismatch for fidelity calculation.")

    try:
        inner_product: complex
        if isinstance(vec1_backend, list): # PurePythonBackend
            vec1_conj = [v.conjugate() for v in vec1_backend]
            inner_product = sum(v1c * v2 for v1c, v2 in zip(vec1_conj, vec2_backend))
        else: # CuPyBackendWrapper
            inner_product_cp = backend.dot(vec1_backend.conj(), vec2_backend)
            inner_product = backend._scalar_or_array(inner_product_cp) # 确保转换为 Python 标量
            
        fidelity_raw = abs(inner_product)**2
        # [健壮性改进] 裁剪保真度到 [0.0, 1.0] 范围，处理浮点误差
        fidelity_clipped = backend.clip(fidelity_raw, 0.0, 1.0)

        return float(fidelity_clipped)
    except Exception as e:
        _public_api_logger.critical(f"API: Fidelity calculation failed: {e}", exc_info=True)
        raise RuntimeError(f"Fidelity calculation failed: {e}") from e

def get_measurement_probabilities(state: 'QuantumState') -> List[float]:
    """
    [健壮性改进版] 从一个量子态中获取所有计算基的测量概率。

    Args:
        state (QuantumState):
            要获取测量概率的量子态。

    Returns:
        List[float]:
            一个包含所有测量概率的 Python 列表。
        
    Raises:
        TypeError: 如果输入 `state` 不是 QuantumState 实例。
        RuntimeError: 如果底层计算失败。
    """
    # [健壮性改进] 输入验证
    if not isinstance(state, QuantumState):
        _public_api_logger.error(f"API: Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    
    _public_api_logger.debug(f"API: Request to get measurement probabilities for {state.num_qubits}-qubit state. Mode: '{state._simulation_mode}'.")
    
    try:
        probabilities = state.get_probabilities()
        _public_api_logger.debug(f"API: Retrieved {len(probabilities)} measurement probabilities.")
        return probabilities
        
    except Exception as e:
        _public_api_logger.critical(f"API: An error occurred while getting probabilities: {e}", exc_info=True)
        raise RuntimeError(f"Failed to get probabilities: {e}") from e
def calculate_entanglement_entropy(state: 'QuantumState', qubits_to_partition: List[int]) -> float:
    """
    [公共API] [健壮性改进版] 计算指定子系统的冯·诺依曼纠缠熵 S(ρ_A) = -Tr(ρ_A log₂(ρ_A))。

    此API函数是访问量子态纠缠信息的官方、安全入口。它会根据需要触发
    惰性求值的量子态的展开计算，然后返回指定子系统的纠缠熵。

    工作流程:
    1.  对输入参数 `state` 和 `qubits_to_partition` 进行严格的类型和值验证。
    2.  将所有复杂的计算逻辑完全委托给 `QuantumState` 对象的内部方法
        `_partial_trace` 和后端的数学函数。
    3.  返回一个标准的 Python `float` 类型的结果。

    Args:
        state (QuantumState):
            要分析的量子态对象。
        qubits_to_partition (List[int]):
            一个整数列表，定义了要计算熵的子系统 `A` 所包含的量子比特。

    Returns:
        float:
            计算出的冯·诺依曼熵（一个非负实数）。

    Raises:
        TypeError: 如果输入参数类型不正确。
        ValueError: 如果 `qubits_to_partition` 无效（如为空、包含重复或越界索引）。
        RuntimeError: 如果底层计算（如部分迹或特征值分解）失败。
    """
    _public_api_logger.debug(f"API: calculate_entanglement_entropy called for N={state.num_qubits}, partition={qubits_to_partition}.")

    # --- 步骤 1: 严格的输入验证 ---
    if not isinstance(state, QuantumState):
        _public_api_logger.error(f"API: Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
        raise TypeError(f"Input 'state' must be a QuantumState instance.")
    
    if not isinstance(qubits_to_partition, list) or not all(isinstance(q, int) and 0 <= q < state.num_qubits for q in qubits_to_partition):
        _public_api_logger.error(f"API: 'qubits_to_partition' must be a non-empty list of valid qubit indices, got {qubits_to_partition}.")
        raise ValueError("qubits_to_partition must be a non-empty list of valid qubit indices.")
    if not qubits_to_partition:
        _public_api_logger.error(f"API: 'qubits_to_partition' cannot be empty.")
        raise ValueError("'qubits_to_partition' cannot be empty.")
    if len(set(qubits_to_partition)) != len(qubits_to_partition):
        _public_api_logger.error(f"API: 'qubits_to_partition' contains duplicate qubit indices: {qubits_to_partition}.")
        raise ValueError("qubits_to_partition contains duplicate qubit indices.")

    try:
        # --- 步骤 2: 将计算委托给 QuantumState 的内部方法 ---
        # 调用内部的 _partial_trace 方法，它会负责触发展开（如果需要）
        qubits_to_trace_out = [q for q in range(state.num_qubits) if q not in qubits_to_partition]
        rho_A = state._partial_trace(qubits_to_trace_out)
        
        # --- 步骤 3: 计算特征值 ---
        eigenvalues = state._backend.eigvalsh(rho_A)
        
        # --- 步骤 4: 计算冯·诺依曼熵 ---
        entropy = 0.0
        for eigenvalue in eigenvalues:
            eigenvalue_float = float(eigenvalue)
            eigenvalue_clipped = state._backend.clip(eigenvalue_float, 0.0, 1.0)
            
            if eigenvalue_clipped > 1e-12:
                entropy -= eigenvalue_clipped * state._backend.log2(eigenvalue_clipped)
        
        return float(entropy)

    except Exception as e:
        _public_api_logger.critical(f"API: An error occurred during entanglement entropy calculation: {e}", exc_info=True)
        # 避免重复包装 RuntimeError
        if isinstance(e, RuntimeError):
            raise e
        raise RuntimeError(f"Entanglement entropy calculation failed: {e}") from e
def get_marginal_probability(state: 'QuantumState', qubit_index: int) -> Tuple[float, float]:
    """
    [健壮性改进版] 获取指定单个量子比特的边际概率分布 (P(0), P(1))。

    Args:
        state (QuantumState):
            要分析的量子态。
        qubit_index (int):
            目标量子比特的索引。

    Returns:
        Tuple[float, float]:
            一个元组 `(prob_0, prob_1)`，分别代表测量该比特
            得到 0 和 1 的概率。

    Raises:
        TypeError: 如果输入 `state` 或 `qubit_index` 的类型不正确。
        ValueError: 如果 `qubit_index` 无效。
        RuntimeError: 如果底层计算失败。
    """
    _public_api_logger.debug(f"API: Request to get marginal probability for qubit {qubit_index} on N={state.num_qubits} state. Mode: '{state._simulation_mode}'.")

    # [健壮性改进] 输入验证
    if not isinstance(state, QuantumState):
        _public_api_logger.error(f"API: Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < state.num_qubits):
        _public_api_logger.error(f"API: Invalid 'qubit_index' {qubit_index} for N={state.num_qubits} state.")
        raise ValueError(f"Invalid qubit index: {qubit_index}.")
        
    try:
        result = state.get_marginal_probabilities(qubit_index)
        if result is None:
            _public_api_logger.critical(f"API: Internal marginal probability calculation failed for qubit {qubit_index}.")
            raise RuntimeError("Internal marginal probability calculation failed.")
        _public_api_logger.debug(f"API: Retrieved marginal probability for qubit {qubit_index}: {result}.")
        return result
        
    except Exception as e:
        _public_api_logger.critical(f"API: An error occurred while getting marginal probability for qubit {qubit_index}: {e}", exc_info=True)
        raise RuntimeError(f"Marginal probability calculation failed for qubit {qubit_index}: {e}") from e

def apply_readout_noise(probabilities: List[float], readout_errors: Dict[int, float]) -> List[float]:
    """
    [健壮性改进版] 对一个理想的概率分布应用简化的、独立的读出错误。

    Args:
        probabilities (List[float]):
            理想的测量概率分布，其长度必须是 2^N。
        readout_errors (Dict[int, float]):
            一个字典，将量子比特索引映射到其读出错误率 (0.0 到 1.0 之间)。
            例如: `{0: 0.01, 1: 0.015}`。

    Returns:
        List[float]:
            应用读出错误后的概率分布。

    Raises:
        TypeError: 如果 `probabilities` 或 `readout_errors` 类型不正确。
        ValueError: 如果 `probabilities` 长度不是 2^N，或 `readout_errors` 包含无效值。
    """
    _public_api_logger.debug("API: Applying readout noise to probabilities.")

    # [健壮性改进] 输入验证
    if not isinstance(probabilities, list) or not all(isinstance(p, (float, int)) for p in probabilities):
        _public_api_logger.error(f"API: 'probabilities' must be a list of numeric values, got {type(probabilities).__name__}.")
        raise TypeError("'probabilities' must be a list of numeric values.")
    if not isinstance(readout_errors, dict) or not all(isinstance(k, int) and isinstance(v, (float, int)) and 0.0 <= v <= 1.0 for k, v in readout_errors.items()):
        _public_api_logger.error(f"API: 'readout_errors' must be a dictionary mapping int to float in [0,1], got {readout_errors}.")
        raise TypeError("'readout_errors' must be a dictionary mapping int to float in [0,1].")

    if not probabilities:
        return []
    
    num_states = len(probabilities)
    try:
        num_qubits = int(math.log2(num_states))
        if (1 << num_qubits) != num_states:
            raise ValueError
    except (ValueError, TypeError):
        _public_api_logger.error(f"API: Length of 'probabilities' list ({num_states}) must be a power of 2.")
        raise ValueError("Length of probabilities list must be a power of 2.")

    if not readout_errors:
        _public_api_logger.debug("API: 'readout_errors' is empty, returning original probabilities.")
        return list(probabilities) # 返回拷贝

    noisy_probs = [0.0] * num_states

    for i, p_ideal_raw in enumerate(probabilities):
        p_ideal = max(0.0, min(1.0, float(p_ideal_raw))) # 裁剪以防浮点误差
        if p_ideal < 1e-12: # 忽略接近零的概率
            continue

        for j in range(num_states):
            prob_flip_i_to_j = 1.0
            for q in range(num_qubits):
                ideal_bit = (i >> q) & 1
                read_bit = (j >> q) & 1
                error_rate = readout_errors.get(q, 0.0) # 如果没有为该比特定义错误，则为 0

                if not (0.0 <= error_rate <= 1.0): # 再次验证错误率
                    _public_api_logger.warning(f"API: Invalid readout error rate {error_rate} for qubit {q}. Clamping to [0,1].")
                    error_rate = max(0.0, min(1.0, error_rate))
                
                if ideal_bit == read_bit:
                    prob_flip_i_to_j *= (1.0 - error_rate)
                else:
                    prob_flip_i_to_j *= error_rate
            noisy_probs[j] += p_ideal * prob_flip_i_to_j
            
    # [健壮性改进] 最终归一化和裁剪
    total_noisy_prob = sum(noisy_probs)
    if not math.isclose(total_noisy_prob, 1.0, abs_tol=1e-7):
        _public_api_logger.warning(f"API: Total noisy probability {total_noisy_prob:.12f} is not 1.0. Renormalizing.")
        if total_noisy_prob > 1e-12:
            normalized_noisy_probs = [p / total_noisy_prob for p in noisy_probs]
        else: # 如果总和接近0，则所有都设为0
            normalized_noisy_probs = [0.0] * num_states
    else:
        normalized_noisy_probs = noisy_probs

    # 裁剪所有概率到 [0, 1]
    final_clipped_probs = [max(0.0, min(1.0, p)) for p in normalized_noisy_probs]

    return final_clipped_probs

# --- VQA, 动力学, 分析 API ---
def calculate_hamiltonian_expectation_value(state: 'QuantumState', hamiltonian: 'Hamiltonian') -> float:
    """
    [健壮性改进版] 计算给定哈密顿量在当前量子态下的期望值。

    Args:
        state (QuantumState):
            要计算期望值的量子态。
        hamiltonian (Hamiltonian):
            一个 `List[PauliString]`，表示要计算期望值的哈密顿量。

    Returns:
        float:
            哈密顿量的期望值（一个实数）。

    Raises:
        TypeError: 如果输入类型不正确。
        ValueError: 如果哈密顿量定义与量子态的比特数不匹配，或哈密顿量非厄米。
        RuntimeError: 如果底层计算失败。
    """
    _public_api_logger.debug(f"API: calculate_hamiltonian_expectation_value called for N={state.num_qubits}. Mode: '{state._simulation_mode}'.")

    # [健壮性改进] 输入验证
    if not isinstance(state, QuantumState):
        _public_api_logger.error(f"API: Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    
    if not isinstance(hamiltonian, list) or not all(isinstance(ps, PauliString) for ps in hamiltonian):
        _public_api_logger.error(f"API: Input 'hamiltonian' must be a list of PauliString objects, got {type(hamiltonian).__name__}.")
        raise TypeError("Input 'hamiltonian' must be a list of PauliString objects.")

    max_hamiltonian_qubit = -1
    for ps_index, ps in enumerate(hamiltonian):
        # 验证 PauliString 实例的有效性 (__post_init__ 已经处理了)
        # 检查哈密顿量是否厄米 (系数必须为实数)
        if abs(ps.coefficient.imag) > 1e-9:
            _public_api_logger.error(f"API: Hamiltonian term {ps_index} ('{ps}') has a complex coefficient ({ps.coefficient}). Hamiltonian must be Hermitian (real coefficients).")
            raise ValueError(f"Hamiltonian must be Hermitian (real coefficients), but found complex coefficient {ps.coefficient} in term '{ps}'.")

        if ps.pauli_map:
            max_hamiltonian_qubit = max(max_hamiltonian_qubit, max(ps.pauli_map.keys()))
    
    if max_hamiltonian_qubit >= state.num_qubits:
        _public_api_logger.error(f"API: Hamiltonian contains operators on qubit {max_hamiltonian_qubit}, which is out of range for a {state.num_qubits}-qubit state.")
        raise ValueError(
            f"Hamiltonian contains operators on qubit {max_hamiltonian_qubit}, which is out of range for a "
            f"{state.num_qubits}-qubit state. All qubit indices in the Hamiltonian must be < {state.num_qubits}."
        )

    _public_api_logger.info(f"API: Calculating expectation value for Hamiltonian with {len(hamiltonian)} terms on a {state.num_qubits}-qubit state. Mode: '{state._simulation_mode}'.")
    
    try:
        return state.get_hamiltonian_expectation(hamiltonian)
    except Exception as e:
        _public_api_logger.critical(f"API: An error occurred while calculating Hamiltonian expectation value: {e}", exc_info=True)
        raise RuntimeError(f"Hamiltonian expectation calculation failed: {e}") from e

def build_trotter_step_circuit(hamiltonian: 'Hamiltonian', time_step: float) -> 'QuantumCircuit':
    """
    [健壮性改进版] 构建一个 Suzuki-Trotter 分解的单步量子线路。

    Args:
        hamiltonian (Hamiltonian): 哈密顿量，一个 `List[PauliString]` 对象。
        time_step (float): 时间步长 `dt`。

    Returns:
        QuantumCircuit: 构建的 Trotter 步线路。

    Raises:
        TypeError: 如果 `hamiltonian` 或 `time_step` 类型不正确。
        ValueError: 如果 `hamiltonian` 非厄米，或包含不支持的 Pauli 字符串。
    """
    _public_api_logger.debug(f"API: build_trotter_step_circuit called for {len(hamiltonian)} terms with dt={time_step}.")

    # [健壮性改进] 输入验证
    if not isinstance(hamiltonian, list) or (hamiltonian and not all(isinstance(ps, PauliString) for ps in hamiltonian)):
        _public_api_logger.error(f"API: 'hamiltonian' must be a list of PauliString objects, got {type(hamiltonian).__name__}.")
        raise TypeError("Hamiltonian must be a list of PauliString objects.")
    if not isinstance(time_step, (float, int)):
        _public_api_logger.error(f"API: 'time_step' must be a numeric value, got {type(time_step).__name__}.")
        raise TypeError("time_step must be a numeric value.")
            
    max_qubit_idx = -1
    for ps_index, ps in enumerate(hamiltonian):
        if abs(ps.coefficient.imag) > 1e-9:
            _public_api_logger.error(f"API: Hamiltonian term {ps_index} ('{ps}') has a complex coefficient ({ps.coefficient}). Hamiltonian must be Hermitian (real coefficients).")
            raise ValueError(f"Hamiltonian must be Hermitian (real coefficients), but found complex coefficient {ps.coefficient} in term '{ps}'.")
        if ps.pauli_map:
            max_qubit_idx = max(max_qubit_idx, max(ps.pauli_map.keys()))
    num_qubits = max_qubit_idx + 1 if max_qubit_idx >= 0 else 0

    try:
        return AlgorithmBuilders.build_trotter_step_circuit(hamiltonian, time_step)
    except Exception as e:
        _public_api_logger.critical(f"API: Failed to build Trotter step circuit: {e}", exc_info=True)
        raise RuntimeError(f"Failed to build Trotter step circuit: {e}") from e

def calculate_entanglement_entropy(self, qubits_to_partition: List[int]) -> float:
        """
        [健壮性改进版] [v1.5.8 builtins fix] 计算指定子系统的冯·诺依曼纠缠熵 S(ρ_A) = -Tr(ρ_A log₂(ρ_A))。
        此版本修复了因错误调用 self.builtins 或 self._backend.builtins 而导致的 AttributeError。
        """
        log_prefix = f"QuantumState.calculate_entanglement_entropy(Partition={qubits_to_partition})"
        self._internal_logger.debug(f"[{log_prefix}] Starting entropy calculation.")

        # --- 1. 输入验证 ---
        if not isinstance(qubits_to_partition, list) or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits_to_partition):
            self._internal_logger.error(f"[{log_prefix}] 'qubits_to_partition' must be a non-empty list of valid qubit indices, got {qubits_to_partition}.")
            raise ValueError("qubits_to_partition must be a non-empty list of valid qubit indices.")
        if not qubits_to_partition:
            self._internal_logger.error(f"[{log_prefix}] 'qubits_to_partition' cannot be empty.")
            raise ValueError("'qubits_to_partition' cannot be empty.")
        
        if len(set(qubits_to_partition)) != len(qubits_to_partition):
            self._internal_logger.error(f"[{log_prefix}] 'qubits_to_partition' contains duplicate qubit indices: {qubits_to_partition}.")
            raise ValueError("qubits_to_partition contains duplicate qubit indices.")

        try:
            # --- 2. 执行部分迹，获取约化密度矩阵 ---
            qubits_to_trace_out = [q for q in range(self.num_qubits) if q not in qubits_to_partition]
            
            rho_A = self._partial_trace(qubits_to_trace_out)
            
            # [健壮性改进] 检查 rho_A 的形状是否正确
            # --- [核心修复] ---
            if (hasattr(rho_A, 'shape') and rho_A.shape[0] != rho_A.shape[1]) or \
                (isinstance(rho_A, list) and rho_A and len(rho_A) != len(rho_A[0])):
                shape_str = rho_A.shape if hasattr(rho_A, 'shape') else (len(rho_A), len(rho_A[0]))
            # --- [修复结束] ---
                self._internal_logger.error(f"[{log_prefix}] Partial trace did not result in a square matrix. Shape: {shape_str}.")
                raise RuntimeError("Partial trace did not result in a square matrix.")

            # --- 3. 计算特征值 ---
            eigenvalues = self._backend.eigvalsh(rho_A)
            
            # --- 4. 计算冯·诺依曼熵 ---
            entropy = 0.0
            
            for eigenvalue in eigenvalues:
                eigenvalue_float = float(eigenvalue)
                # [健 robust性改进] 裁剪特征值到正数，避免对负数取对数（浮点误差可能导致微小负值）
                eigenvalue_clipped = self._backend.clip(eigenvalue_float, 0.0, 1.0)
                
                if eigenvalue_clipped > 1e-12: # 避免对零取对数
                    entropy -= eigenvalue_clipped * self._backend.log2(eigenvalue_clipped)
            
            return float(entropy)

        except Exception as e:
            self._internal_logger.critical(f"[{log_prefix}] An error occurred during entropy calculation: {e}", exc_info=True)
            raise RuntimeError(f"Entanglement entropy calculation failed: {e}") from e
def get_bloch_vector(state: 'QuantumState', qubit_index: int) -> Tuple[float, float, float]:
    """
    [健壮性改进版] 计算指定单个量子比特的布洛赫矢量 (rx, ry, rz)。

    Args:
        state (QuantumState):
            要分析的量子态。
        qubit_index (int):
            目标量子比特的索引。

    Returns:
        Tuple[float, float, float]: 布洛赫矢量 (rx, ry, rz)。

    Raises:
        TypeError: 如果输入类型不正确。
        ValueError: 如果 `qubit_index` 无效。
        RuntimeError: 如果底层计算失败。
    """
    _public_api_logger.debug(f"API: get_bloch_vector called for N={state.num_qubits} qubit={qubit_index}. Mode: '{state._simulation_mode}'.")

    # [健壮性改进] 输入验证
    if not isinstance(state, QuantumState):
        _public_api_logger.error(f"API: Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    if not isinstance(qubit_index, int) or not (0 <= qubit_index < state.num_qubits):
        _public_api_logger.error(f"API: Invalid 'qubit_index' {qubit_index} for N={state.num_qubits} state.")
        raise ValueError(f"Invalid qubit index: {qubit_index}.")

    try:
        return state.get_bloch_vector(qubit_index)
    except Exception as e:
        _public_api_logger.critical(f"API: An error occurred while getting the Bloch vector for qubit {qubit_index}: {e}", exc_info=True)
        raise RuntimeError(f"Bloch vector calculation failed: {e}") from e


def _generate_random_pure_state(num_qubits: int) -> List[complex]:
    """
    [独立测试辅助函数] 生成一个随机的、归一化的纯态量子态矢量。
    此函数从 nexus_optimizer.py 复制而来，以确保 quantum_core.py 的独立可测试性。
    """
    if not isinstance(num_qubits, int) or num_qubits < 0:
        raise ValueError("num_qubits must be a non-negative integer.")

    if num_qubits == 0:
        return [1.0 + 0.0j]
        
    dim = 1 << num_qubits
    vec = [complex(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)) for _ in range(dim)]
    
    norm_sq = sum(amp.real**2 + amp.imag**2 for amp in vec)

    if norm_sq < 1e-12:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Generated random vector norm ({norm_sq:.2e}) is close to zero. "
            "Returning a deterministic ground state |0...0⟩ instead."
        )
        result_vec = [0.0 + 0.0j] * dim
        result_vec[0] = 1.0 + 0.0j
        return result_vec

    norm = math.sqrt(norm_sq)
    normalized_vec = [amp / norm for amp in vec]
    return normalized_vec

# ========================================================================
# --- 10. 独立测试块 ---
# ========================================================================

if __name__ == '__main__':
    """
    [健壮性改进版] 当此脚本被直接执行时，运行此处的测试代码。
    此版本完全适配惰性求值和双模式架构，并新增了对模式切换的验证。
    """
    
    # --- 步骤 1: 配置一个基础的日志记录器 ---
    logging.basicConfig(
        level=logging.INFO, # 默认INFO，减少日志量，如果需要调试请改为DEBUG
        format='%(asctime)s - [%(levelname)s] - (%(name)s) - %(message)s'
    )
    
    print("\n" + "="*80)
    print("--- Running NEXUS QUANTUM CORE Standalone Self-Tests (New Lazy Architecture) ---")
    print("="*80 + "\n")

    # --- 步骤 2: 定义一个通用的测试结果打印函数 ---
    def run_test(test_name: str, test_function: callable, backend_choice: str = "pure_python"):
        """运行一个测试函数，并捕获结果和时间。"""
        # 记录原始配置，确保测试后恢复
        original_core_config = copy.deepcopy(_core_config)
        original_log_level = logging.getLogger().level

        try:
            # 配置核心库，为测试准备环境
            # 提高 MAX_QUBITS 以支持 QPE 等算法可能需要更多比特的测试
            # 调整 PARALLEL_COMPUTATION_QUBIT_THRESHOLD 以便测试并行性
            configure_quantum_core({
                "BACKEND_CHOICE": backend_choice, 
                "MAX_QUBITS": 32, 
                "PARALLEL_COMPUTATION_QUBIT_THRESHOLD": 8 
            }) 
            # 将日志级别设为DEBUG以便在测试失败时获取详细信息
            logging.getLogger().setLevel(logging.DEBUG) 

            print(f"[*] Running Test: {test_name} (Backend: {backend_choice})...")
            start_time = time.perf_counter()
            test_function()
            duration = (time.perf_counter() - start_time) * 1000
            print(f"✅ PASSED: {test_name} (Backend: {backend_choice}) (in {duration:.2f} ms)\n")
            return True
        except NotImplementedError as e:
            duration = (time.perf_counter() - start_time) * 1000
            print(f"⚠️ SKIPPED: {test_name} (Backend: {backend_choice}) (after {duration:.2f} ms)")
            print(f"    REASON: Function not implemented for this backend/version. {e}\n")
            return True
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            print(f"❌ FAILED: {test_name} (Backend: {backend_choice}) (after {duration:.2f} ms)")
            print(f"    ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc(limit=5)
            print("")
            return False
        finally:
            # 恢复原始配置和日志级别
            configure_quantum_core(original_core_config)
            logging.getLogger().setLevel(original_log_level)


    # --- 步骤 3: 编写各个功能的测试函数 ---

    def test_state_and_circuit_creation():
        """
        测试 QuantumState 和 QuantumCircuit 的基本创建和惰性初始化。
        """
        _public_api_logger.info("    - Testing QuantumState(3) lazy creation...")
        state = create_quantum_state(3)
        assert state.num_qubits == 3, "State should have 3 qubits"
        assert state._simulation_mode == 'statevector', "Initial state should be in statevector mode."
        assert not state._is_cache_valid, "Cache should be invalid after creation."
        assert len(state.circuit.instructions) == 0, "Internal circuit should be empty."
        _public_api_logger.info("    - QuantumState(3) lazy creation successful.")

        _public_api_logger.info("    - Testing adding gate invalidates cache...")
        state.h(0)
        assert not state._is_cache_valid, "Cache should be invalid after adding gate."
        assert len(state.circuit.instructions) == 1, "Internal circuit should have 1 instruction."
        _public_api_logger.info("    - Adding gate invalidates cache successful.")

        _public_api_logger.info("    - Testing 0-qubit state creation...")
        state_0q = create_quantum_state(0)
        assert state_0q.num_qubits == 0, "0-qubit state should have 0 qubits"
        assert state_0q._simulation_mode == 'statevector', "0-qubit state should be in statevector mode."
        _public_api_logger.info("    - 0-qubit state creation successful.")

        _public_api_logger.info("    - Testing invalid input for state creation (negative qubits)...")
        try:
            create_quantum_state(-1)
            assert False, "Should have raised ValueError for negative qubits"
        except ValueError: pass
        _public_api_logger.info("    - Invalid input test successful.")
    def test_parallel_computation_correctness():
        """
        [新增] 明确验证并行计算内核的正确性。
        此测试通过对比并行与串行计算的最终态矢量来确保结果一致。
        """
        # 选择一个能触发并行Hadamard内核的比特数
        # 必须大于等于 PARALLEL_COMPUTATION_QUBIT_THRESHOLD (在测试配置中设为8)
        num_qubits_for_parallel_test = 12 
        
        _public_api_logger.info(f"    - Testing parallel computation correctness for {num_qubits_for_parallel_test} qubits...")
        
        # 准备一个复杂的初始态
        qc_prep = QuantumCircuit(num_qubits_for_parallel_test)
        for i in range(num_qubits_for_parallel_test):
            qc_prep.ry(i, 0.1 * (i + 1))
        
        initial_state_template = create_quantum_state(num_qubits_for_parallel_test)
        initial_state_complex = run_circuit_on_state(initial_state_template, qc_prep)

        # --- 路径 A: 强制并行计算 ---
        _public_api_logger.info("      - Running Path A (Parallel)...")
        enable_parallelism() # 确保并行已启用
        assert _parallel_enabled, "Parallelism failed to enable for Path A."
        
        state_parallel = copy.deepcopy(initial_state_complex)
        state_parallel.h(0) # 这个H门应该会触发并行内核

        # [*** 最终核心修复 ***]
        # 显式触发计算。这将调用 _expand_to_statevector，内部会根据并行设置选择路径。
        get_measurement_probabilities(state_parallel)
        
        assert state_parallel._cached_statevector_entity is not None, "Parallel path failed to create cached entity."
        vec_parallel = state_parallel._cached_statevector_entity._state_vector
        disable_parallelism() # 清理

        # --- 路径 B: 强制串行计算 ---
        _public_api_logger.info("      - Running Path B (Serial)...")
        # 确保并行已禁用
        assert not _parallel_enabled, "Parallelism should be disabled for Path B."

        state_serial = copy.deepcopy(initial_state_complex)
        # 手动将阈值调高，确保不触发并行
        original_threshold = _core_config["PARALLEL_COMPUTATION_QUBIT_THRESHOLD"]
        _core_config["PARALLEL_COMPUTATION_QUBIT_THRESHOLD"] = 100 
        state_serial.h(0) # 这个H门现在应该会触发串行内核
        
        # [*** 最终核心修复 ***]
        # 同样需要显式触发计算
        get_measurement_probabilities(state_serial)
        _core_config["PARALLEL_COMPUTATION_QUBIT_THRESHOLD"] = original_threshold # 恢复阈值
        
        assert state_serial._cached_statevector_entity is not None, "Serial path failed to create cached entity."
        vec_serial = state_serial._cached_statevector_entity._state_vector

        # --- 断言 ---
        _public_api_logger.info("      - Comparing results...")
        fidelity = calculate_fidelity_sv(vec_parallel, vec_serial, state_parallel._backend)
        
        assert math.isclose(fidelity, 1.0, abs_tol=1e-9), \
            f"Parallel computation correctness test FAILED. Fidelity between parallel and serial results is {fidelity:.12f}, not 1.0."
        _public_api_logger.info("    - Parallel computation correctness test successful.")
    def test_kernel_vs_matrix_consistency():
        """
        [最终架构修复版 - 2025-10-25] 验证 _StateVectorEntity 的优化内核路径与
        通用矩阵构建路径的结果一致性。

        此测试的核心是确保 `_apply_..._kernel_sv` 的直接态矢量操作，
        与其通过 `get_effective_unitary` 计算出的全局酉矩阵作用在态矢量上的结果完全相同。

        测试流程:
        1.  准备一个随机初始态。
        2.  路径 A (内核路径): 直接在一个实体上调用优化内核方法 (e.g., `_apply_toffoli_kernel_sv`)。
        3.  路径 B (矩阵路径):
            a. 创建一个只包含当前测试门的临时 QuantumCircuit。
            b. 调用 `get_effective_unitary` 为这个电路生成一个“黄金标准”的全局酉矩阵。
               此函数会正确处理原子门和复合宏的展开。
            c. 将这个酉矩阵通过矩阵乘法应用到另一个实体上。
        4.  计算路径 A 和路径 B 最终态矢量之间的保真度，验证其是否接近 1.0。
        """
        num_qubits = 4 # 选择一个不大不小的比特数
        _public_api_logger.info(f"    - Testing kernel vs. matrix consistency for {num_qubits} qubits...")

        # 获取当前测试配置下的后端实例
        backend = create_quantum_state(num_qubits)._backend
        
        # 定义要测试的门及其参数。这些门都在 _atomic_kernel_gates 集合中，有优化内核。
        gates_to_test = [
            ('h', (0,)),
            ('cnot', (0, 1)),
            ('rx', (2, math.pi / 3)),
            ('cp', (1, 3, math.pi / 4)),
            ('toffoli', (0, 1, 2)),
        ]
        
        for gate_name, args in gates_to_test:
            _public_api_logger.info(f"      - Verifying gate: {gate_name}{args}...")
            
            # 准备一个随机初始态
            initial_vec = _generate_random_pure_state(num_qubits)
            # 确保初始态是后端兼容的格式
            initial_vec_backend = backend._ensure_cupy_array(initial_vec) if isinstance(backend, CuPyBackendWrapper) else copy.deepcopy(initial_vec)

            # --- 路径 A: 内核路径 ---
            entity_kernel = _StateVectorEntity(num_qubits, backend)
            entity_kernel._state_vector = copy.deepcopy(initial_vec_backend)
            
            # 手动调用内核方法
            kernel_method_name = f'_apply_{gate_name}_kernel_sv'
            assert hasattr(entity_kernel, kernel_method_name), f"Kernel method {kernel_method_name} not found in _StateVectorEntity."
            kernel_func = getattr(entity_kernel, kernel_method_name)
            kernel_func(*args)
            vec_after_kernel = entity_kernel._state_vector
            
            # --- [*** 最终核心修复 - 2025-10-25 v7 ***] ---
            # 路径 B: 通用矩阵构建路径 (使用 get_effective_unitary 作为黄金标准)
            
            # 1. 构建一个只包含当前门的临时电路
            temp_qc = QuantumCircuit(num_qubits)
            temp_qc.add_gate(gate_name, *args)
            
            # 2. 计算这个单门电路的全局有效酉矩阵
            #    get_effective_unitary 会正确处理宏展开（对于复合宏）或直接构建（对于原子宏）
            global_unitary = get_effective_unitary(temp_qc)
            
            # 3. 将这个矩阵作用于初始态矢量
            entity_matrix = _StateVectorEntity(num_qubits, backend)
            entity_matrix._state_vector = copy.deepcopy(initial_vec_backend)
            entity_matrix._apply_global_unitary(global_unitary)
            final_vec_matrix = entity_matrix._state_vector
            # --- [*** 修复结束 ***] ---

            # --- 断言 ---
            # 计算两个最终态矢量的保真度
            fidelity = calculate_fidelity_sv(vec_after_kernel, final_vec_matrix, backend)
            assert math.isclose(fidelity, 1.0, abs_tol=1e-9), \
                f"Kernel vs. Matrix consistency test FAILED for gate '{gate_name}'. Fidelity: {fidelity:.12f}"

        _public_api_logger.info("    - All kernel vs. matrix consistency tests successful.")
    
    
    def test_gate_application_and_circuit_run():
        """
        测试门操作的惰性应用和线路执行后的最终状态（贝尔态）。
        """
        _public_api_logger.info("    - Creating initial |00> state...")
        initial_state = create_quantum_state(2) # 用于传递给 run_circuit_on_state

        _public_api_logger.info("    - Lazily building Bell state preparation circuit (H on q0, CNOT(0,1))...")
        bell_qc = QuantumCircuit(2, description="Bell State Prep Test")
        bell_qc.h(0)
        bell_qc.cnot(0, 1) 
        
        # 验证初始状态的内部电路是空的
        assert len(initial_state.circuit.instructions) == 0, "Initial state's circuit should be empty before run_circuit_on_state."

        _public_api_logger.info("    - Running the circuit on the state (this will trigger expansion)...")
        final_state = run_circuit_on_state(initial_state, bell_qc)

        # 验证原始状态未被修改 (因为 run_circuit_on_state 内部做了深拷贝)
        assert len(initial_state.circuit.instructions) == 0, "Original state's circuit should remain empty (deep copy)."
        assert not initial_state._is_cache_valid, "Original state's cache should remain invalid."

        _public_api_logger.info("    - Verifying final state probabilities (this will trigger expansion for final_state)...")
        probabilities = get_measurement_probabilities(final_state) # 强制 final_state 进行计算
        expected_probs = [0.5, 0.0, 0.0, 0.5]
        
        assert len(probabilities) == 4, f"Expected 4 probabilities, but got {len(probabilities)}"
        assert all(math.isclose(p_actual, p_expected, abs_tol=1e-9) 
                for p_actual, p_expected in zip(probabilities, expected_probs)), \
            f"Bell state probabilities are incorrect. Expected {expected_probs}, got {probabilities}"
        
        assert final_state.is_valid(), "Final state (Bell state) should be valid."
        _public_api_logger.info("    - Bell state verification successful.")
    def test_mode_switching_and_noise():
        """
        测试从 statevector 到 density_matrix 模式的自动切换，并验证噪声是否生效。
        """
        _public_api_logger.info("    - Defining a simple hardware noise model...")
        fake_calib = {
            'gates': {
                'h': {'duration': 50e-9, 'error_rate': 0.1} # 10% 去极化错误
            }
        }
        noise_model = PrebuiltNoiseModels.HardwareBackend(fake_calib)
        
        # 准备初始态和线路
        state_lazy = create_quantum_state(1)
        qc_noise_test_circuit = QuantumCircuit(1)
        qc_noise_test_circuit.h(0)
        
        # --- 路径 A - 理想模拟 ---
        _public_api_logger.info("    - Path A: Running ideal simulation...")
        ideal_state = run_circuit_on_state(state_lazy, qc_noise_test_circuit, noise_model=None)
        # 触发展开并获取其密度矩阵表示
        dm_ideal = ideal_state.density_matrix
        
        # --- 路径 B - 带噪模拟 ---
        _public_api_logger.info("    - Path B: Running noisy simulation (should trigger mode switch)...")
        noisy_state = run_circuit_on_state(state_lazy, qc_noise_test_circuit, noise_model=noise_model)
        
        # 验证模式切换
        assert noisy_state._simulation_mode == 'density_matrix', "Noise should force mode switch to density_matrix."
        assert noisy_state._density_matrix is not None, "Density matrix should be initialized after noise."

        # 获取噪声态的密度矩阵
        dm_noisy = noisy_state._density_matrix 

        # --- 验证纯度 ---
        # 纯度 Tr(ρ^2)
        purity_ideal = ideal_state._backend.trace(ideal_state._backend.dot(dm_ideal, dm_ideal)).real
        purity_noisy = noisy_state._backend.trace(noisy_state._backend.dot(dm_noisy, dm_noisy)).real

        _public_api_logger.info(f"    - Ideal state purity: {purity_ideal:.6f} (Expected: ≈1.0)")
        _public_api_logger.info(f"    - Noisy state purity: {purity_noisy:.6f} (Expected: <1.0)")

        assert math.isclose(purity_ideal, 1.0, abs_tol=1e-9), "Ideal state after H-gate should be pure (purity=1)."
        assert purity_noisy < 1.0, f"Noisy state purity {purity_noisy} should be less than 1.0 (mixed state)."
        assert not ideal_state._backend.allclose(dm_ideal, dm_noisy), "Noisy state should differ from the ideal state."

        _public_api_logger.info("    - Mode switching and noise application test successful.")
    def test_qft_and_inverse_qft():
        """
        测试量子傅里叶变换 (QFT) 及其逆操作 (IQFT) 的正确性。
        """
        num_qubits = 3
        target_state_int = 5  # Corresponds to |101> for qubits |q2 q1 q0>
        
        _public_api_logger.info(f"    - Preparing initial state |{target_state_int:0{num_qubits}b}⟩...")
        initial_state_for_qft = create_quantum_state(num_qubits)
        initial_state_for_qft.x(0)
        initial_state_for_qft.x(2)
        
        _public_api_logger.info("    - Lazily building QFT and IQFT circuits using AlgorithmBuilders...")
        qft_circuit = AlgorithmBuilders.build_qft_circuit(num_qubits, inverse=False)
        iqft_circuit = AlgorithmBuilders.build_qft_circuit(num_qubits, inverse=True)

        # 合并QFT和IQFT到单个电路中进行测试
        full_qft_iqft_circuit = QuantumCircuit(num_qubits, description="QFT-IQFT Cycle Test")
        full_qft_iqft_circuit.instructions.extend(qft_circuit.instructions)
        full_qft_iqft_circuit.instructions.extend(iqft_circuit.instructions)

        _public_api_logger.info("    - Applying combined QFT+IQFT (this will trigger expansion)...")
        
        # [*** 核心修复：将电路应用到正确的初始态 ***]
        # final_state = run_circuit_on_state(create_quantum_state(num_qubits), full_qft_iqft_circuit) # 错误行
        final_state = run_circuit_on_state(initial_state_for_qft, full_qft_iqft_circuit) # 正确行

        _public_api_logger.info("    - Verifying final state (this will trigger expansion for final_state)...")
        final_probabilities = get_measurement_probabilities(final_state) 
        
        assert math.isclose(final_probabilities[target_state_int], 1.0, abs_tol=1e-7), \
            f"QFT->IQFT failed. Probability of initial state |{target_state_int:0{num_qubits}b}⟩ " \
            f"is {final_probabilities[target_state_int]:.6f}, expected 1.0"
        
        for i, prob in enumerate(final_probabilities):
            if i != target_state_int:
                assert math.isclose(prob, 0.0, abs_tol=1e-7), \
                    f"QFT->IQFT failed. Probability of non-target state |{i:0{num_qubits}b}⟩ " \
                    f"is {prob:.6f}, expected 0.0"
        _public_api_logger.info("    - QFT and IQFT cycle test successful.")

    def test_partial_trace_and_entropy():
        """
        [测试用例] [v2.0 健壮性与双模式增强版] 全面测试部分迹和冯·诺依曼纠缠熵的计算。
        [v2.1 API调用修复] 此版本修正了对 `calculate_entanglement_entropy` 的调用方式。
        [v2.2 混合态数据修复] 修正了混合态测试场景中手动构建的密度矩阵的数据。
        """
        _public_api_logger.info("    - Starting comprehensive partial trace and entropy tests...")

        # --- 场景 1: 最大纠缠态 (贝尔态) ---
        _public_api_logger.info("      - Scene 1: Testing on a 2-qubit Bell state (|Φ+⟩)...")
        
        state_bell = create_quantum_state(2)
        state_bell.h(0)
        state_bell.cnot(0, 1)

        _public_api_logger.info("        - Calculating partial trace over qubit 1 (triggers expansion)...")
        rho_q0 = state_bell._partial_trace([1])
        
        expected_rho_q0 = [[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]]
        assert state_bell._backend.allclose(rho_q0, expected_rho_q0, atol=1e-9), \
            f"Partial trace on Bell state is incorrect. Expected {expected_rho_q0}, got {rho_q0}"
        _public_api_logger.info("        - Partial trace on Bell state is correct (result is I/2).")

        _public_api_logger.info("        - Calculating entanglement entropy for one qubit of the Bell state...")
        entropy_q0 = calculate_entanglement_entropy(state_bell, [0])
        expected_entropy_bell = 1.0 # log₂(2)
        
        assert math.isclose(entropy_q0, expected_entropy_bell, abs_tol=1e-9), \
            f"Entropy of entangled qubit incorrect. Expected {expected_entropy_bell}, got {entropy_q0}"
        _public_api_logger.info(f"        - Entanglement entropy on Bell state is correct (S ≈ {entropy_q0:.4f}).")

        # --- 场景 2: 可分乘积态 ---
        _public_api_logger.info("      - Scene 2: Testing on a 2-qubit product state |0+⟩...")

        product_state = create_quantum_state(2)
        product_state.h(1) # State is now |0⟩ ⊗ |+⟩

        rho_prod_q0 = product_state._partial_trace([1])
        
        expected_rho_prod_q0 = [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]
        assert product_state._backend.allclose(rho_prod_q0, expected_rho_prod_q0, atol=1e-9), \
            f"Partial trace on product state is incorrect. Expected {expected_rho_prod_q0}, got {rho_prod_q0}"
        _public_api_logger.info("        - Partial trace on product state is correct (result is |0⟩⟨0|).")

        entropy_prod_q0 = calculate_entanglement_entropy(product_state, [0])
        expected_entropy_prod = 0.0
        
        assert math.isclose(entropy_prod_q0, expected_entropy_prod, abs_tol=1e-9), \
            f"Entropy of product state qubit incorrect. Expected {expected_entropy_prod}, got {entropy_prod_q0}"
        _public_api_logger.info(f"        - Entanglement entropy on product state is correct (S ≈ {entropy_prod_q0:.4f}).")

        # --- 场景 3: 多比特 GHZ 态 ---
        _public_api_logger.info("      - Scene 3: Testing on a 3-qubit GHZ state...")
        
        state_ghz = create_quantum_state(3)
        state_ghz.h(0)
        state_ghz.cnot(0, 1)
        state_ghz.cnot(0, 2) # Creates (|000⟩ + |111⟩)/√2

        entropy_ghz_q0 = calculate_entanglement_entropy(state_ghz, [0])
        expected_entropy_ghz = 1.0
        
        assert math.isclose(entropy_ghz_q0, expected_entropy_ghz, abs_tol=1e-9), \
            f"Entropy of one qubit in GHZ state incorrect. Expected {expected_entropy_ghz}, got {entropy_ghz_q0}"
        _public_api_logger.info(f"        - Entanglement entropy on GHZ state is correct (S ≈ {entropy_ghz_q0:.4f}).")

        # --- 场景 4: 混合态部分迹 (Density Matrix Mode) ---
        _public_api_logger.info("      - Scene 4: Testing partial trace on a pre-defined mixed state...")
        
        state_mixed_dm = create_quantum_state(2)
        state_mixed_dm._switch_to_density_matrix_mode(reason="Manual setup for test")
        
        # [!!! 核心修复 !!!]
        # 手动构建一个正确的混合态密度矩阵：ρ = 0.5 * |Φ+⟩⟨Φ+| + 0.5 * |00⟩⟨00|
        # ρ = 0.75*|00⟩⟨00| + 0.25*|00⟩⟨11| + 0.25*|11⟩⟨00| + 0.25*|11⟩⟨11|
        # 基矢顺序: |00⟩, |01⟩, |10⟩, |11⟩
        dm_data = [
            [0.75, 0.00, 0.00, 0.25],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.25, 0.00, 0.00, 0.25]
        ]
        if isinstance(state_mixed_dm._backend, CuPyBackendWrapper):
            state_mixed_dm._density_matrix = state_mixed_dm._backend._cp.array(dm_data, dtype=complex)
        else:
            state_mixed_dm._density_matrix = [[complex(v) for v in row] for row in dm_data]

        # b) 计算对 q1 的部分迹
        #    Tr_1(ρ) = 0.5 * Tr_1(|Φ+⟩⟨Φ+|) + 0.5 * Tr_1(|00⟩⟨00|)
        #            = 0.5 * (I/2) + 0.5 * |0⟩⟨0|
        #            = [[0.25, 0], [0, 0.25]] + [[0.5, 0], [0, 0]] = [[0.75, 0], [0, 0.25]]
        rho_mixed_q0 = state_mixed_dm._partial_trace([1])
        
        expected_rho_mixed_q0 = [[0.75, 0.0], [0.0, 0.25]]
        
        assert state_mixed_dm._backend.allclose(rho_mixed_q0, expected_rho_mixed_q0, atol=1e-9), \
            f"Partial trace on mixed state is incorrect. Expected {expected_rho_mixed_q0}, got {rho_mixed_q0}"
        _public_api_logger.info("        - Partial trace on mixed state is correct.")
        
        # --- 场景 5: 错误处理验证 ---
        _public_api_logger.info("      - Scene 5: Testing error handling for invalid inputs...")
        
        try:
            calculate_entanglement_entropy(state_bell, []) # 空列表
            assert False, "Should have raised ValueError for empty qubits_to_partition list."
        except ValueError: pass
        
        try:
            calculate_entanglement_entropy(state_bell, [0, 0]) # 重复索引
            assert False, "Should have raised ValueError for duplicate qubit indices."
        except ValueError: pass
        
        try:
            calculate_entanglement_entropy(state_bell, [99]) # 越界索引
            assert False, "Should have raised ValueError for out-of-bounds qubit index."
        except ValueError: pass

        _public_api_logger.info("        - Error handling for invalid inputs is correct.")
        
        _public_api_logger.info("    - Partial trace and entropy tests fully completed and successful.")
    
    
    
    
    def test_hamiltonian_expectation():
        """
        [测试用例] [v2.0 健壮性与双模式增强版] 全面测试哈密顿量期望值的计算。

        此测试用例覆盖了以下关键场景：
        1.  **纯态验证 (Statevector Mode)**:
            -   在一个简单的、已知结果的纯态（`|+⟩`）上，验证 `X` 和 `Z` 算子的期望值，
                确保 `calculate_hamiltonian_expectation_value` 在 `statevector` 模式下
                能正确触发展开并计算。
        2.  **混合态验证 (Density Matrix Mode)**:
            -   手动创建一个已知的混合态密度矩阵（例如，一个部分去极化的态）。
            -   在该混合态上计算期望值，并与手动计算的理论值进行比较，以验证
                `calculate_hamiltonian_expectation_value` 在 `density_matrix` 模式下的
                正确性。
        3.  **多比特哈密顿量验证**:
            -   在一个纠缠态（贝尔态）上，测试一个包含多比特Pauli串（如 `ZZ`）的
                哈密顿量的期望值。
        4.  **错误处理验证**:
            -   验证当传入非厄米哈密顿量（包含复数系数）或比特索引越界的
                PauliString 时，函数是否能按预期抛出 `ValueError`。

        此测试完全依赖于公共API，确保了其行为与用户所经历的一致。
        """
        _public_api_logger.info("    - Starting comprehensive Hamiltonian expectation value tests...")

        # --- 场景 1: 纯态期望值 (Statevector Mode) ---
        _public_api_logger.info("      - Scene 1: Testing on a pure state |+⟩...")
        
        # a) 准备 |+> 态
        state_plus = create_quantum_state(1)
        state_plus.h(0) # H|0> = |+>

        # b) 定义哈密顿量 H = 1.0 * X_0
        hamiltonian_x = [PauliString(coefficient=1.0, pauli_map={0: 'X'})]
        
        # c) 计算期望值 <+|X|+>，期望结果为 1.0
        #    此调用会触发 state_plus 的惰性展开
        _public_api_logger.info("        - Calculating <+|X|+> (triggers expansion)...")
        expected_energy_x = 1.0
        calculated_energy_x = calculate_hamiltonian_expectation_value(state_plus, hamiltonian_x)
        
        assert math.isclose(calculated_energy_x, expected_energy_x, abs_tol=1e-9), \
            f"Hamiltonian expectation value incorrect for <+|X|+>. Expected {expected_energy_x}, got {calculated_energy_x}"
        _public_api_logger.info("        - Verification successful for <+|X|+>.")

        # d) 定义哈密顿量 H = 1.0 * Z_0
        hamiltonian_z = [PauliString(coefficient=1.0, pauli_map={0: 'Z'})]
        
        # e) 计算期望值 <+|Z|+>，期望结果为 0.0
        _public_api_logger.info("        - Calculating <+|Z|+>...")
        expected_energy_z = 0.0
        calculated_energy_z = calculate_hamiltonian_expectation_value(state_plus, hamiltonian_z)
        
        assert math.isclose(calculated_energy_z, expected_energy_z, abs_tol=1e-9), \
            f"Hamiltonian expectation value incorrect for <+|Z|+>. Expected {expected_energy_z}, got {calculated_energy_z}"
        _public_api_logger.info("        - Verification successful for <+|Z|+>.")

        # --- 场景 2: 混合态期望值 (Density Matrix Mode) ---
        _public_api_logger.info("      - Scene 2: Testing on a mixed state...")

        # a) 手动构建一个混合态：ρ = 0.8 |0><0| + 0.2 |1><1|
        #    这是一个部分去极化的态
        p = 0.2
        state_mixed = create_quantum_state(1)
        # 强制切换到密度矩阵模式，并手动设置密度矩阵
        state_mixed._switch_to_density_matrix_mode(reason="Manual setup for test")
        
        dm_data = [[1.0 - p, 0.0], [0.0, p]]
        if isinstance(state_mixed._backend, CuPyBackendWrapper):
            state_mixed._density_matrix = state_mixed._backend._cp.array(dm_data, dtype=complex)
        else:
            state_mixed._density_matrix = [[complex(v) for v in row] for row in dm_data]
        
        # b) 在此混合态上计算 <Z> = Tr(Zρ)
        #    Tr(Zρ) = Tr([[1,0],[0,-1]] * [[1-p,0],[0,p]]) = Tr([[1-p,0],[0,-p]]) = 1-p-p = 1-2p
        _public_api_logger.info("        - Calculating Tr(Z * ( (1-p)|0><0| + p|1><1| ))...")
        expected_energy_mixed_z = 1.0 - 2.0 * p
        calculated_energy_mixed_z = calculate_hamiltonian_expectation_value(state_mixed, hamiltonian_z)
        
        assert math.isclose(calculated_energy_mixed_z, expected_energy_mixed_z, abs_tol=1e-9), \
            f"Expectation on mixed state incorrect for <Z>. Expected {expected_energy_mixed_z}, got {calculated_energy_mixed_z}"
        _public_api_logger.info("        - Verification successful for mixed state <Z>.")
        
        # --- 场景 3: 多比特哈密顿量 ---
        _public_api_logger.info("      - Scene 3: Testing a multi-qubit Hamiltonian on a Bell state...")
        
        # a) 准备贝尔态 |Φ+⟩ = (|00⟩ + |11⟩)/√2
        state_bell = create_quantum_state(2)
        state_bell.h(0)
        state_bell.cnot(0, 1)

        # b) 定义哈密顿量 H = Z_0 ⊗ Z_1
        hamiltonian_zz = [PauliString(coefficient=1.0, pauli_map={0: 'Z', 1: 'Z'})]

        # c) 计算期望值 <Φ+|ZZ|Φ+>。对于贝尔态，Z0和Z1的测量结果总是相关的，所以ZZ=1
        _public_api_logger.info("        - Calculating <Φ+|ZZ|Φ+>...")
        expected_energy_zz = 1.0
        calculated_energy_zz = calculate_hamiltonian_expectation_value(state_bell, hamiltonian_zz)

        assert math.isclose(calculated_energy_zz, expected_energy_zz, abs_tol=1e-9), \
            f"Expectation for <Φ+|ZZ|Φ+> incorrect. Expected {expected_energy_zz}, got {calculated_energy_zz}"
        _public_api_logger.info("        - Verification successful for multi-qubit Hamiltonian.")

        # --- 场景 4: 错误处理验证 ---
        _public_api_logger.info("      - Scene 4: Testing error handling for invalid Hamiltonians...")

        # a) 测试非厄米哈密顿量（复数系数）
        try:
            non_hermitian_h = [PauliString(coefficient=1j, pauli_map={0: 'X'})]
            calculate_hamiltonian_expectation_value(state_plus, non_hermitian_h)
            assert False, "Should have raised ValueError for non-Hermitian Hamiltonian."
        except ValueError as e:
            assert "Hermitian" in str(e), "Error message for non-Hermitian check is incorrect."
            _public_api_logger.info("        - Correctly raised ValueError for non-Hermitian Hamiltonian.")

        # b) 测试比特索引越界
        try:
            out_of_bounds_h = [PauliString(coefficient=1.0, pauli_map={99: 'X'})] # state_plus只有1个比特
            calculate_hamiltonian_expectation_value(state_plus, out_of_bounds_h)
            assert False, "Should have raised ValueError for out-of-bounds qubit index."
        except ValueError as e:
            assert "out of range" in str(e), "Error message for out-of-bounds check is incorrect."
            _public_api_logger.info("        - Correctly raised ValueError for out-of-bounds qubit index.")
            
        _public_api_logger.info("    - Hamiltonian expectation value tests fully completed and successful.")

    def test_classical_control_flow():
        """
        测试经典控制流 (conditional operations)。
        """
        _public_api_logger.info("    - Building a circuit with measurement and a conditional gate...")
        
        # 构建一个完整的、可重用的测试电路。
        qc_test_circuit = QuantumCircuit(2, description="Classical Control Test")
        qc_test_circuit.h(0)
        qc_test_circuit.measure(0, classical_register_index=0)
        qc_test_circuit.x(1, condition=(0, 1))

        _public_api_logger.info("    - Running the circuit multiple times to observe both outcomes...")
        num_trials = 20
        outcomes_00 = 0
        outcomes_11 = 0
        
        for i in range(num_trials):
            # [*** 核心修复：在此处为每次试验重新播种随机数生成器 ***]
            # 使用 os.urandom 可以提供高质量的随机种子，确保每次试验的随机性。
            random.seed(os.urandom(16)) 

            # 每次都从全新的初始态开始
            initial_state_for_trial = create_quantum_state(2)
            
            # 正确的流程是使用 `run_circuit_on_state` 来执行整个电路。
            final_state = run_circuit_on_state(initial_state_for_trial, qc_test_circuit)

            # 触发展开并获取最终概率
            final_probs = get_measurement_probabilities(final_state)
            
            # 由于测量坍缩，最终状态应该是纯态 |00⟩ 或 |11⟩
            is_00_state = math.isclose(final_probs[0], 1.0, abs_tol=1e-7)
            is_11_state = math.isclose(final_probs[3], 1.0, abs_tol=1e-7)
            
            assert is_00_state or is_11_state, \
                f"Invalid final state in trial {i+1}. Probabilities: {final_probs}. Expected pure |00> or |11>."
            
            if is_00_state: outcomes_00 += 1
            if is_11_state: outcomes_11 += 1
        
        _public_api_logger.info(f"    - Results after {num_trials} trials: {outcomes_00} times -> |00⟩, {outcomes_11} times -> |11⟩.")
        assert outcomes_00 + outcomes_11 == num_trials, "Total number of outcomes does not match number of trials."
        assert outcomes_00 > 0 and outcomes_11 > 0, \
            "Expected both |00> and |11> outcomes to occur in 20 trials (this is a stochastic test and might rarely fail)."
        
        _public_api_logger.info("    - Classical control flow logic verified successfully.")

    def test_coherent_noise_model():
        """
        [最终正确性修复版] 测试相干噪声模型是否正确地修改了门参数。
        此版本通过确保两条比较路径使用完全相同的计算方法来消除数值精度差异。
        """
        _public_api_logger.info("    - Creating a deterministic coherent noise model (std=0, mean=0.1)...")
        noise_model = PrebuiltNoiseModels.CoherentGateError(angle_error_mean=0.1, angle_error_std=0.0)
        
        base_angle = 0.5
        error_angle = 0.1
        
        # --- 路径 A: 自动噪声路径 ---
        # 运行一个带噪声的线路，这将自动切换到密度矩阵模式并应用有误差的门。
        initial_state_A = create_quantum_state(1)
        qc1 = QuantumCircuit(1)
        qc1.rx(0, base_angle)
        final_state_noisy = run_circuit_on_state(initial_state_A, qc1, noise_model=noise_model)
        dm_noisy = final_state_noisy._density_matrix
        
        assert final_state_noisy._simulation_mode == 'density_matrix', "Noise should force mode switch to density_matrix."
        assert dm_noisy is not None, "Density matrix should be initialized after noise."

        # --- 路径 B: 手动黄金标准路径 ---
        # 手动创建一个密度矩阵实体，并直接在其上应用一个带有正确误差角度的门。
        
        # 1. 创建一个初始的 |0><0| 密度矩阵实体
        manual_entity = _DensityMatrixEntity(1, initial_state_A._backend)
        
        # 2. 创建一个只包含 RX(base_angle + error_angle) 的临时电路
        qc_manual = QuantumCircuit(1)
        qc_manual.rx(0, base_angle + error_angle)

        # 3. 直接在实体上运行这个电路。这将调用与路径 A 完全相同的内核或后备路径。
        manual_entity.run_circuit_on_entity(qc_manual)
        dm_manual = manual_entity._density_matrix
        
        # --- 验证 ---
        _public_api_logger.info("    - Verifying that the final density matrices from both paths are identical...")
        
        are_matrices_close = final_state_noisy._backend.allclose(
            dm_noisy, 
            dm_manual, 
            atol=1e-9
        )

        if not are_matrices_close:
            max_diff = 0.0
            if isinstance(dm_noisy, list):
                diff_matrix = [[abs(n - m) for n, m in zip(row_n, row_m)] for row_n, row_m in zip(dm_noisy, dm_manual)]
                max_diff = max(max(row) for row in diff_matrix)
            else:
                diff_matrix = dm_noisy - dm_manual
                max_diff = final_state_noisy._backend.abs(diff_matrix).max().item()
            
            assert False, (
                "Coherent noise model did not produce the same state as a manually adjusted ideal gate. "
                f"Max absolute difference: {max_diff:.2e}"
            )

        _public_api_logger.info("    - Coherent noise model test successful.")

    def test_hardware_noise_model():
        """
        测试硬件噪声模型是否引入了非相干噪声，将纯态变为混合态。
        """
        _public_api_logger.info("    - Defining a mock hardware calibration data...")
        fake_calib = {
            'qubits': {
                0: {'T1': 10e-6, 'T2': 20e-6} # 10μs T1, 20μs T2
            },
            'gates': {
                'h': {'duration': 50e-9, 'error_rate': 0.001} # 50ns H-gate with 0.1% error
            }
        }
        noise_model = PrebuiltNoiseModels.HardwareBackend(fake_calib)
        
        # 准备初始态和电路
        state_lazy = create_quantum_state(1)
        qc_noise_test_circuit = QuantumCircuit(1)
        qc_noise_test_circuit.h(0)
        
        # 路径 A - 理想模拟 (保持惰性)
        ideal_state = run_circuit_on_state(state_lazy, qc_noise_test_circuit, noise_model=None)
        
        # 路径 B - 带噪模拟 (强制切换为密度矩阵)
        noisy_state = run_circuit_on_state(state_lazy, qc_noise_test_circuit, noise_model=noise_model)
        
        assert noisy_state._simulation_mode == 'density_matrix', "Noise should force mode switch to density_matrix."
        assert noisy_state._density_matrix is not None, "Density matrix should be initialized after noise."

        # 计算纯度 Tr(ρ^2)
        dm_ideal = ideal_state.density_matrix # 会触发展开
        dm_noisy = noisy_state._density_matrix 

        purity_ideal = ideal_state._backend.trace(ideal_state._backend.dot(dm_ideal, dm_ideal)).real
        purity_noisy = noisy_state._backend.trace(noisy_state._backend.dot(dm_noisy, dm_noisy)).real

        _public_api_logger.info(f"    - Ideal state purity: {purity_ideal:.6f} (Expected: ≈1.0)")
        _public_api_logger.info(f"    - Noisy state purity: {purity_noisy:.6f} (Expected: <1.0)")

        assert math.isclose(purity_ideal, 1.0, abs_tol=1e-9), "Ideal state after H-gate should be pure (purity=1)."
        assert purity_noisy < 1.0, f"Noisy state purity {purity_noisy} should be less than 1.0 (mixed)."
        assert purity_noisy > 0.5, f"Noisy state purity {purity_noisy} is unexpectedly low, check noise parameters."
        assert not ideal_state._backend.allclose(dm_ideal, dm_noisy), "Noisy state should differ from the ideal state."

        _public_api_logger.info("    - Hardware noise model test successful.")

    def test_readout_noise_application():
        """
        测试 `apply_readout_noise` 公共 API 函数的计算是否正确。
        """
        _public_api_logger.info("    - Defining an ideal probability distribution for |00>...")
        ideal_probs = [1.0, 0.0, 0.0, 0.0] # 2 qubits

        _public_api_logger.info("    - Defining readout errors: q0=0.1, q1=0.2...")
        readout_errors = {0: 0.1, 1: 0.2} 
        
        _public_api_logger.info("    - Applying readout noise...")
        noisy_probs = apply_readout_noise(ideal_probs, readout_errors)
        
        # 如果理想是 |00> (p_00=1), 错误率 q0=0.1, q1=0.2
        # P(read 00 | ideal 00) = (1-0.1)*(1-0.2) = 0.9 * 0.8 = 0.72
        # P(read 01 | ideal 00) = (1-0.1)*0.2    = 0.9 * 0.2 = 0.18
        # P(read 10 | ideal 00) = 0.1*(1-0.2)    = 0.1 * 0.8 = 0.08
        # P(read 11 | ideal 00) = 0.1*0.2        = 0.1 * 0.2 = 0.02
        expected_noisy = [0.72, 0.08, 0.18, 0.02] # Note: Order is |00>, |01>, |10>, |11> (q1q0)
        
        _public_api_logger.info(f"    - Expected noisy distribution: {expected_noisy}")
        _public_api_logger.info(f"    - Actual noisy distribution:   {[f'{p:.2f}' for p in noisy_probs]}")

        assert len(noisy_probs) == len(expected_noisy), "Length of noisy probabilities mismatch."
        assert all(math.isclose(p_actual, p_expected, abs_tol=1e-9) 
                   for p_actual, p_expected in zip(noisy_probs, expected_noisy)), \
            f"Readout noise calculation incorrect. Expected {expected_noisy}, got {noisy_probs}"
            
        assert math.isclose(sum(noisy_probs), 1.0, abs_tol=1e-9), \
            f"Total probability after readout noise is {sum(noisy_probs)}, should be 1.0"
            
        _public_api_logger.info("    - Readout noise application test successful.")

    def test_new_gates_iswap_ecr():
        """
        测试新添加的 iSWAP 和 ECR 门的正确性，并验证 ECR 与其逆操作 ECRdg 的互逆性。
        """
        num_qubits = 2
        
        # --- iSWAP Gate Test ---
        _public_api_logger.info(f"    - Testing iSWAP gate on {num_qubits} qubits...")
        state_iswap = create_quantum_state(num_qubits)
        state_iswap.x(0) # Prepare |01> state (q1=0, q0=1)
        state_iswap.iswap(0, 1) # Expected i|10> state (q1=1, q0=0)
        
        probs_iswap = get_measurement_probabilities(state_iswap)
        
        _public_api_logger.debug(f"      iSWAP: Final probabilities: {probs_iswap}")
        
        assert math.isclose(probs_iswap[2], 1.0, abs_tol=1e-9), \
            f"iSWAP test FAILED: Expected |10> probability to be 1.0, but got {probs_iswap[2]:.6f} (full distribution: {probs_iswap})"
        
        # 验证相位
        state_iswap._expand_to_statevector()
        final_vec_iswap = state_iswap._cached_statevector_entity._state_vector
        is_list_backend_iswap = isinstance(final_vec_iswap, list)
        amplitude_10 = final_vec_iswap[2] if is_list_backend_iswap else final_vec_iswap[2].item()
        
        _public_api_logger.debug(f"      iSWAP: Final amplitude for |10>: {amplitude_10}")

        assert math.isclose(amplitude_10.real, 0.0, abs_tol=1e-9) and math.isclose(amplitude_10.imag, 1.0, abs_tol=1e-9), \
            f"iSWAP test FAILED: Expected amplitude for |10> to be 1j, but got {amplitude_10} (real={amplitude_10.real:.6f}, imag={amplitude_10.imag:.6f})"
        _public_api_logger.info("    - iSWAP gate test successful.")

        # --- ECR Gate Test (宏与矩阵定义一致性验证) ---
        _public_api_logger.info(f"    - Testing ECR gate macro vs matrix definition consistency on {num_qubits} qubits...")

         # 1. 获取 ECR 宏展开后的酉矩阵
        ecr_macro_circuit = QuantumCircuit(num_qubits, description="ECR Macro Decomposition")
        ecr_macro_circuit.ecr(0, 1) # 调用 ECR 宏

        unitary_from_macro = get_effective_unitary(ecr_macro_circuit)
        
        # 2. 获取 _get_local_op_for_gate 中的 ECR 矩阵定义
        # 需要一个临时的 _StateVectorEntity 实例来调用其内部方法
        temp_state_for_backend_ops = create_quantum_state(num_qubits)
        temp_entity_for_matrix_def = _StateVectorEntity(num_qubits, temp_state_for_backend_ops._backend)
        unitary_from_matrix_def = temp_entity_for_matrix_def._get_local_op_for_gate('ecr', [0, 1])

        # 3. 比较这两个矩阵
        if unitary_from_matrix_def is None:
            raise AssertionError("ECR matrix definition in _get_local_op_for_gate returned None.")

        _public_api_logger.debug(f"      ECR: Unitary from macro:\n{unitary_from_macro}")
        _public_api_logger.debug(f"      ECR: Unitary from matrix def:\n{unitary_from_matrix_def}")
        
        assert temp_state_for_backend_ops._backend.allclose(unitary_from_macro, unitary_from_matrix_def, atol=1e-9), \
            f"ECR macro decomposition DOES NOT match its matrix definition.\n" \
            f"  Macro Result:\n{unitary_from_macro}\n" \
            f"  Matrix Def:\n{unitary_from_matrix_def}"
        
        _public_api_logger.info("    - ECR gate macro vs matrix definition consistency successful.")

        # --- ECR and ECRdg Inverse Test ---
        _public_api_logger.info(f"    - Verifying that ECR and ECRdg macros are inverses on {num_qubits} qubits...")
        
        # 1. 创建一个非平凡的初始态（仅用于逻辑参考，最终比较的是酉矩阵）
        state_initial_ref = create_quantum_state(num_qubits)
        state_initial_ref.h(0)
        state_initial_ref.ry(1, 0.5 * math.pi / 3) 
        
        # 2. 构建 ECR + ECRdg 的宏展开电路
        inverse_test_circuit_macro = QuantumCircuit(num_qubits, description="ECR-ECRdg Inverse Macro Sequence")
        inverse_test_circuit_macro.ecr(0, 1) # 调用 ECR 宏
        inverse_test_circuit_macro.ecrdg(0, 1) # 调用 ECRdg 宏

        # 3. 计算整个序列的有效酉矩阵 (应该接近单位矩阵)
        final_unitary_from_sequence = get_effective_unitary(inverse_test_circuit_macro)
        identity_matrix = state_initial_ref._backend.eye(1 << num_qubits)

        _public_api_logger.debug(f"      ECR-ECRdg: Final unitary from sequence:\n{final_unitary_from_sequence}")
        _public_api_logger.debug(f"      ECR-ECRdg: Expected identity matrix:\n{identity_matrix}")

        assert state_initial_ref._backend.allclose(final_unitary_from_sequence, identity_matrix, atol=1e-9), \
            f"ECR and ECRdg macro sequence DID NOT result in an identity matrix.\n" \
            f"  Actual Final Unitary:\n{final_unitary_from_sequence}\n" \
            f"  Expected Identity:\n{identity_matrix}"
            
        _public_api_logger.info("    - ECR-ECRdg inverse macro test successful.")
    
    
    def test_qpe_circuit_builder():
        """
        [最终正确性修复版] 测试 Quantum Phase Estimation (QPE) 线路构建器。
        使用 U = RZ(pi/2) 估算 |1> 的相位 (期望 φ = 0.125)。
        """
        _public_api_logger.info(f"    - Testing QPE circuit builder...")
        num_counting_qubits = 3
        target_qubit_idx = 0 

        # 定义酉算子 U = RZ(pi/2)
        u_phi = math.pi / 2.0
        u_circuit_1bit = QuantumCircuit(1) # 作用于一个局部比特0
        u_circuit_1bit.rz(0, u_phi)
        
        # [*** 最终核心修复：定义正确的计数比特顺序 ***]
        # 定义计数寄存器使用的全局比特索引。
        # 约定：列表按小端序排列，即 counting_qubits[0] 是最低位比特 q_0,
        # counting_qubits[1] 是 q_1, 以此类推。
        counting_qubits = [1, 2, 3] # [q_0, q_1, q_2] 的全局索引
        
        qpe_qc = AlgorithmBuilders.build_qpe_circuit(
            counting_qubits=counting_qubits,
            target_qubits=[target_qubit_idx],
            unitary_circuit=u_circuit_1bit
        )
        
        # 准备QPE的初始态：目标比特 |1>，计数比特 |0...0>
        initial_qpe_state = create_quantum_state(qpe_qc.num_qubits)
        initial_qpe_state.x(target_qubit_idx) # 制备目标比特 q0 为 |1>
        
        _public_api_logger.info("    - Running QPE circuit (this will trigger expansion)...")
        final_qpe_state = run_circuit_on_state(initial_qpe_state, qpe_qc)
        
        # 物理学: U|ψ⟩ = e^(2πiφ)|ψ⟩。对于 RZ(θ)|1⟩ = e^(iθ/2)|1⟩，我们有 2πφ = θ/2。
        # 当 θ = π/2 时，2πφ = π/4，所以 φ = 1/8 = 0.125。
        # 对于 3 个计数比特，φ 的二进制小数是 0.001。
        # IQFT(no swaps) 会将反向编码的相位 |100> (二进制) 变换为 |001> (二进制)。
        # 我们的受控U门序列恰好就是这样反向编码相位的。
        # 因此，我们期望在计数寄存器上测量到 |001⟩。
        # 对应的十进制整数是 1。
        expected_counting_outcome_idx = 1 # |001⟩
        
        # 测量计数比特的概率分布
        qubits_to_trace_out_for_counting = [q for q in range(qpe_qc.num_qubits) if q not in counting_qubits]
        rho_counting_qubits = final_qpe_state._partial_trace(qubits_to_trace_out_for_counting)
        
        counting_probs_raw = final_qpe_state._backend.diag(rho_counting_qubits)
        
        # 鲁棒地将后端数组/列表转换为Python浮点数列表
        if hasattr(counting_probs_raw, 'tolist'):
            counting_probs_list = counting_probs_raw.tolist()
        elif isinstance(counting_probs_raw, list):
            counting_probs_list = counting_probs_raw
        else: # e.g., 1D CuPy array
            counting_probs_list = [val.item() for val in counting_probs_raw]
            
        counting_probs = [p.real for p in counting_probs_list]
        
        # 归一化以处理浮点误差
        total_counting_prob = sum(counting_probs)
        if total_counting_prob > 1e-12:
            counting_probs = [p / total_counting_prob for p in counting_probs]
        
        prob_at_expected_outcome = counting_probs[expected_counting_outcome_idx]
        
        _public_api_logger.info(f"    - Expected phase φ=0.125. Expect high prob at outcome {expected_counting_outcome_idx} (|001>).")
        _public_api_logger.info(f"    - Probability at expected outcome: {prob_at_expected_outcome:.4f}")
        
        assert prob_at_expected_outcome > 0.9, \
            f"QPE for RZ(pi/2) on |1> failed. Expected high prob at counting outcome {expected_counting_outcome_idx}, got {prob_at_expected_outcome}"

        # --- QPE 边界条件测试 ---
        _public_api_logger.info("    - Testing QPE with invalid inputs (should raise errors)...")
        try:
            AlgorithmBuilders.build_qpe_circuit(counting_qubits=[], target_qubits=[0], unitary_circuit=u_circuit_1bit)
            assert False, "Should have raised ValueError for empty counting_qubits."
        except ValueError: pass

        try:
            AlgorithmBuilders.build_qpe_circuit(counting_qubits=[1], target_qubits=[], unitary_circuit=u_circuit_1bit)
            assert False, "Should have raised ValueError for empty target_qubits."
        except ValueError: pass
        
        try:
            invalid_u_circuit = QuantumCircuit(2) # 2局部比特
            invalid_u_circuit.swap(0, 1) 
            # 这里的 unitary_circuit 是 2 比特，但 target_qubits 是 [0] (1比特) -> 应该报错
            AlgorithmBuilders.build_qpe_circuit(counting_qubits=[2], target_qubits=[0], unitary_circuit=invalid_u_circuit)
            assert False, "Should have raised ValueError for unitary_circuit.num_qubits mismatch with len(target_qubits)."
        except ValueError: pass

        _public_api_logger.info("    - QPE circuit builder tests successful.")
    
    
    def test_hardware_efficient_ansatz_builder():
        """
        测试 Hardware-Efficient Ansatz 线路构建器。
        """
        _public_api_logger.info(f"    - Testing Hardware-Efficient Ansatz builder...")
        num_qubits = 2
        depth = 1
        parameters = [math.pi/2, math.pi/4, math.pi/2, math.pi/4] # 2 qubits * 2 params/qubit * 1 depth
        
        ansatz_qc = AlgorithmBuilders.build_hardware_efficient_ansatz(
            num_qubits=num_qubits,
            depth=depth,
            parameters=parameters,
            entanglement_type='linear'
        )
        
        # 当 depth=1 时，纠缠层(cnot)不会被添加，因为 d < depth - 1 (0 < 0) 为假。
        expected_instructions = [
            ('ry', 0, parameters[0]),
            ('rz', 0, parameters[1]),
            ('ry', 1, parameters[2]),
            ('rz', 1, parameters[3]),
        ]
        
        assert len(ansatz_qc.instructions) == len(expected_instructions), \
            f"Ansatz circuit length mismatch. Expected {len(expected_instructions)}, got {len(ansatz_qc.instructions)}"
            
        for i, (actual, expected) in enumerate(zip(ansatz_qc.instructions, expected_instructions)):
            assert actual[0] == expected[0], f"Gate name mismatch at instruction {i}. Expected {expected[0]}, got {actual[0]}."
            if actual[0] in ['ry', 'rz']: # 对于浮点数参数，使用 isclose 比较
                assert math.isclose(actual[2], expected[2]), f"Gate angle mismatch at instruction {i}. Expected {expected[2]}, got {actual[2]}."
            else:
                assert actual[1:] == expected[1:], f"Gate args/kwargs mismatch at instruction {i}. Expected {expected[1:]}, got {actual[1:]}."


        # 运行线路验证其有效性
        state = create_quantum_state(num_qubits)
        final_state = run_circuit_on_state(state, ansatz_qc)
        assert final_state.is_valid(), "Hardware-Efficient Ansatz produced an invalid quantum state."
        
        # --- Ansatz 边界条件和错误测试 ---
        _public_api_logger.info("    - Testing Hardware-Efficient Ansatz with invalid inputs (should raise errors)...")
        try:
            # 参数列表长度不匹配
            AlgorithmBuilders.build_hardware_efficient_ansatz(num_qubits=2, depth=1, parameters=[0.0]) # 期望4个参数
            assert False, "Should have raised ValueError for mismatched parameters length."
        except ValueError: pass

        try:
            # 无效的纠缠类型
            AlgorithmBuilders.build_hardware_efficient_ansatz(num_qubits=2, depth=1, parameters=[0]*4, entanglement_type='invalid_type')
            assert False, "Should have raised ValueError for invalid entanglement_type."
        except ValueError: pass

        _public_api_logger.info("    - Hardware-Efficient Ansatz builder test successful.")

    def test_multi_controlled_gates():
        """
        专门测试多控制门 (MCZ, MCX, MCP, MCU) 的正确性，
        并验证优化内核路径与矩阵构建路径的一致性。
        """
        num_qubits = 4
        controls = [0, 1]
        target = 2
        
        # --- Test 1: MCZ (Multi-Controlled-Z) ---
        _public_api_logger.info("    - Testing MCZ gate...")
        state_mcz_kernel = create_quantum_state(num_qubits)
        for q in controls + [target]: state_mcz_kernel.h(q) # Create superposition
        
        # 路径 A: 使用优化内核
        state_mcz_kernel.mcz(controls, target)
        probs_kernel = get_measurement_probabilities(state_mcz_kernel)

        # 路径 B: 使用矩阵构建 (通过 get_effective_unitary)
        state_mcz_matrix = create_quantum_state(num_qubits)
        qc_mcz_matrix = QuantumCircuit(num_qubits)
        for q in controls + [target]: qc_mcz_matrix.h(q)
        qc_mcz_matrix.mcz(controls, target)
        unitary_mcz = get_effective_unitary(qc_mcz_matrix) # 获取完整酉矩阵
        
        # 在原始状态上应用酉矩阵
        state_mcz_matrix._expand_to_statevector()
        state_mcz_matrix._cached_statevector_entity._apply_global_unitary(unitary_mcz) # 直接应用
        probs_matrix = get_measurement_probabilities(state_mcz_matrix)

        assert all(math.isclose(p1, p2, abs_tol=1e-9) for p1, p2 in zip(probs_kernel, probs_matrix)), \
            "MCZ kernel and matrix paths produced different results."
        _public_api_logger.info("    - MCZ test successful.")

        # --- Test 2: MCX (Multi-Controlled-X / Toffoli) ---
        _public_api_logger.info("    - Testing MCX (Toffoli) gate...")
        state_mcx = create_quantum_state(num_qubits)
        state_mcx.x(controls[0])
        state_mcx.x(controls[1]) # Prepare |...1100> state on controls
        state_mcx.mcx(controls, target) # Should flip target to |...1110>
        
        probs_mcx = get_measurement_probabilities(state_mcx)
        # Expected outcome is |...1110> (q3q2q1q0) if controls are q0,q1 and target is q2
        expected_idx = (1 << controls[0]) | (1 << controls[1]) | (1 << target)
        assert math.isclose(probs_mcx[expected_idx], 1.0, abs_tol=1e-9), \
            f"MCX gate failed. Expected prob=1.0 at index {expected_idx}, got {probs_mcx[expected_idx]}"
        _public_api_logger.info("    - MCX test successful.")

        # --- Test 3: MCP (Multi-Controlled-Phase) ---
        _public_api_logger.info("    - Testing MCP gate...")
        state_mcp = create_quantum_state(num_qubits)
        for q in range(num_qubits): state_mcp.h(q) # Equal superposition
        
        angle = math.pi / 4
        state_mcp.mcp(controls, target, angle) # Apply MCP using kernel

        # Verification: build the matrix and compare states
        state_mcp_manual = create_quantum_state(num_qubits)
        for q in range(num_qubits): state_mcp_manual.h(q)
        
        # [*** 最终核心修复 ***]
        # _build_mcp_operator 是 _StateVectorEntity 的方法，不是 QuantumState 的方法。
        # 我们需要先获取或创建一个 entity 实例来调用它。
        state_mcp_manual._expand_to_statevector()
        entity_for_build = state_mcp_manual._cached_statevector_entity
        mcp_matrix = entity_for_build._build_mcp_operator(controls, target, angle)
        
        entity_for_build._apply_global_unitary(mcp_matrix)
        
        state_mcp._expand_to_statevector()
        fidelity = calculate_fidelity_sv(
            state_mcp._cached_statevector_entity._state_vector,
            state_mcp_manual._cached_statevector_entity._state_vector,
            state_mcp._backend
        )
        assert math.isclose(fidelity, 1.0, abs_tol=1e-9), "MCP kernel and matrix paths are inconsistent."
        _public_api_logger.info("    - MCP test successful.")

        # --- Test 4: MCU (Multi-Controlled-U) ---
        _public_api_logger.info("    - Testing MCU gate...")
        state_mcu = create_quantum_state(num_qubits)
        state_mcu.x(controls[0])
        state_mcu.x(controls[1]) # Prepare |...1100> state
        
        u_matrix_example = [[0, 1], [1, 0]] # Local X gate
        state_mcu.mcu(controls, target, u_matrix_example) # Should apply X on target if controls are 1

        # Expected outcome: if controls were 11, target (q2) should flip 0->1
        # This is the same logic as MCX.
        state_mcu_manual_mcx = create_quantum_state(num_qubits)
        state_mcu_manual_mcx.x(controls[0])
        state_mcu_manual_mcx.x(controls[1])
        state_mcu_manual_mcx.mcx(controls, target)
        
        state_mcu._expand_to_statevector()
        state_mcu_manual_mcx._expand_to_statevector()

        fidelity_mcu = calculate_fidelity_sv(
            state_mcu._cached_statevector_entity._state_vector,
            state_mcu_manual_mcx._cached_statevector_entity._state_vector,
            state_mcu._backend
        )
        assert math.isclose(fidelity_mcu, 1.0, abs_tol=1e-9), "MCU kernel with X matrix is inconsistent with MCX gate."
        _public_api_logger.info("    - MCU test successful.")

    def test_crosstalk_model():
        """
        全面测试增强版的 CorrelatedNoise 模型。
        
        此测试覆盖了模型的三个核心特性：
        1. 混合噪声：相干酉替换与非相干通道的结合。
        2. 方向性：为 CNOT(A,B) 和 CNOT(B,A) 定义不同规则。
        3. 通配符：为作用于特定比特上的“任何”门定义通用规则。
        """
        _public_api_logger.info("    - Running comprehensive tests for the enhanced CorrelatedNoise model...")

        # --- 辅助函数，用于测试内部 ---
        def _calculate_purity(state: 'QuantumState') -> float:
            """计算一个量子态的纯度 Tr(ρ^2)。"""
            if state._simulation_mode == 'statevector':
                return 1.0 # 对于纯态，纯度理论上是1
            dm = state._density_matrix
            if dm is None: return 0.0
            purity = state._backend.trace(state._backend.dot(dm, dm)).real
            return float(purity)

        def _calculate_fidelity_density_matrix(state_a: 'QuantumState', state_b: 'QuantumState') -> float:
            """计算两个密度矩阵之间的保真度 F(ρ, σ) = Tr(sqrt(sqrt(ρ) σ sqrt(ρ)))^2。
               对于纯态 ρ = |ψ><ψ|，简化为 F(|ψ><ψ|, σ) = <ψ|σ|ψ>。
               对于这里的测试，都是从纯态开始，然后经过噪声变为混合态，所以可以使用 Tr(ρ_pure * σ_mixed)
            """
            dm_a = state_a.density_matrix # 确保展开并获取密度矩阵
            dm_b = state_b._density_matrix if state_b._simulation_mode == 'density_matrix' else state_b.density_matrix
            if dm_a is None or dm_b is None: return 0.0
            
            # 这里假设其中一个可以是纯态密度矩阵，另一个是混合态。
            # 如果两个都是混合态，标准保真度计算更复杂。
            # For simplicity in this test: We check if they are identical, or if one is pure.
            
            # 如果两个密度矩阵完全相同，保真度就是1
            if state_a._backend.allclose(dm_a, dm_b, atol=1e-9):
                return 1.0
            
            # 否则，如果其中一个是纯态 (例如 dm_a = |psi><psi|)
            # 则 F(|psi><psi|, dm_b) = <psi|dm_b|psi>
            # 需要从 dm_a 还原 |psi>，但这有点复杂。
            # 简单方法是计算 Tr(dm_a * dm_b)
            fidelity_trace_product = state_a._backend.trace(state_a._backend.dot(dm_a, dm_b)).real
            return float(state_a._backend.clip(fidelity_trace_product, 0.0, 1.0))


        # --- 场景 A: 混合噪声（相干替换 + 非相干通道）验证 ---
        _public_api_logger.info("    - Scene A: Verifying mixed coherent replacement and incoherent channels...")

        # a) 定义噪声模型
        replacement_qc = QuantumCircuit(
            3, description="Noisy CNOT(0,1)",
            instructions=[
                ('cnot', 0, 1), # 原始门
                ('rz', 0, 0.02),      # 在控制位上的相干误差
                ('rx', 1, -0.01),     # 在目标位上的相干误差
                ('rz', 2, 0.05)       # 对旁观者q2的相干串扰
            ]
        )
        crosstalk_map_A = {
            'cnot': {
                (0, 1): {
                    'error_model': {
                        'coherent_unitary_replacement': {'circuit': replacement_qc},
                        'incoherent_post_op': [
                            {'channel_type': 'depolarizing', 'target_qubits': [2], 'params': {'probability': 0.001}}
                        ]
                    }
                }
            }
        }
        noise_model_A = PrebuiltNoiseModels.CorrelatedNoise(crosstalk_map_A)

        # b) 路径 1: 使用噪声模型自动应用
        initial_state_A = create_quantum_state(3)
        circuit_to_run_A = QuantumCircuit(3)
        circuit_to_run_A.h(0)
        circuit_to_run_A.cnot(0, 1) # 这个 CNOT 将会被替换并附加噪声
        
        state_noisy_auto = run_circuit_on_state(initial_state_A, circuit_to_run_A, noise_model=noise_model_A)

        # c) 路径 2: 手动构建“黄金标准”路径
        initial_state_B = create_quantum_state(3)
        initial_state_B.h(0)
        initial_state_B._switch_to_density_matrix_mode() # 强制切换到密度矩阵模式，以匹配噪声模型的行为
        
        # 手动运行替换电路
        initial_state_B.run_circuit(replacement_qc)
        # 手动应用非相干通道
        initial_state_B.apply_quantum_channel(
            channel_type='depolarizing', target_qubits=[2], params={'probability': 0.001}
        )

        # d) 验证：两个路径的最终密度矩阵必须几乎完全相同
        assert state_noisy_auto._simulation_mode == 'density_matrix', "噪声模型应强制切换到密度矩阵模式。"
        assert initial_state_B._simulation_mode == 'density_matrix', "黄金标准路径也应处于密度矩阵模式。"
        
        dm_auto = state_noisy_auto._density_matrix
        dm_manual = initial_state_B._density_matrix
        
        assert dm_auto is not None and dm_manual is not None, "两个路径都应生成有效的密度矩阵。"
        
        are_matrices_close = state_noisy_auto._backend.allclose(dm_auto, dm_manual, atol=1e-9)
        assert are_matrices_close, f"自动噪声模型生成的最终态与手动构建的黄金标准不匹配. Max diff: {state_noisy_auto._backend.abs(dm_auto - dm_manual).max()}"
        _public_api_logger.info("    - Scene A: Mixed noise model test successful.")

        # --- 场景 B & C: 方向性和通配符特性验证 ---
        _public_api_logger.info("    - Scene B & C: Verifying directionality and wildcard features...")
        
        # a) 定义一个包含多种规则的地图
        wildcard_map = {
            'cnot': {
                (1, 0): { # 规则只对 CNOT(1,0)
                    'error_model': {'incoherent_post_op': [{'channel_type': 'phase_flip', 'target_qubits': [0], 'params': {'probability': 0.1}}]}
                }
            },
            'any': {
                (2, 3): { # 规则对作用于 (2,3) 的任何双比特门
                    'error_model': {'incoherent_post_op': [{'channel_type': 'bit_flip', 'target_qubits': [2,3], 'params': {'probability': 0.2}}]}
                }
            }
        }
        wildcard_noise_model = PrebuiltNoiseModels.CorrelatedNoise(wildcard_map)
        
        # b) 测试方向性：CNOT(0,1) 应该没有噪声 (不匹配 (1,0))
        noise_for_01 = wildcard_noise_model.get_noise_for_op('cnot', [0, 1])
        assert not noise_for_01, "方向性测试失败: CNOT(0,1) 不应匹配 CNOT(1,0) 的规则。"
        
        # c) 测试方向性：CNOT(1,0) 应该有噪声
        noise_for_10 = wildcard_noise_model.get_noise_for_op('cnot', [1, 0])
        assert 'incoherent_post_op' in noise_for_10 and noise_for_10['incoherent_post_op'][0]['channel_type'] == 'phase_flip', "方向性测试失败: CNOT(1,0) 应匹配其精确规则。"

        # d) 测试通配符：CZ(2,3) 应该匹配 'any' 规则
        noise_for_cz_23 = wildcard_noise_model.get_noise_for_op('cz', [2, 3])
        assert 'incoherent_post_op' in noise_for_cz_23 and noise_for_cz_23['incoherent_post_op'][0]['channel_type'] == 'bit_flip', "通配符测试失败: CZ(2,3) 应匹配 'any' on (2,3) 的规则。"

        # e) 测试通配符：SWAP(2,3) 也应该匹配
        noise_for_swap_23 = wildcard_noise_model.get_noise_for_op('swap', [2, 3])
        assert 'incoherent_post_op' in noise_for_swap_23 and noise_for_swap_23['incoherent_post_op'][0]['channel_type'] == 'bit_flip', "通配符测试失败: SWAP(2,3) 也应匹配 'any' on (2,3) 的规则。"

        # f) 测试优先级：如果 'cnot' 和 'any' 都有匹配，精确的 'cnot' 规则应优先
        priority_map = {
            'cnot': { (0, 1): {'error_model': {'incoherent_post_op': [{'channel_type': 'phase_flip', 'target_qubits': [0], 'params': {'probability': 0.1}}]}}},
            'any': { (0, 1): {'error_model': {'incoherent_post_op': [{'channel_type': 'bit_flip', 'target_qubits': [0], 'params': {'probability': 0.1}}]}}}
        }
        priority_noise_model = PrebuiltNoiseModels.CorrelatedNoise(priority_map)
        noise_for_priority = priority_noise_model.get_noise_for_op('cnot', [0, 1])
        assert noise_for_priority['incoherent_post_op'][0]['channel_type'] == 'phase_flip', "优先级测试失败: 精确的 'cnot' 规则应优先于 'any' 规则。"

        _public_api_logger.info("    - Scene B & C: Directionality and wildcard tests successful.")
        _public_api_logger.info("    - Overall crosstalk model test successful.")
    
    def test_large_scale_simulation_performance(num_qubits: int, backend_choice: str):
        """
        对大规模模拟进行一次性能演示测试。
        """
        _public_api_logger.info(f"--- Performance Demo: {num_qubits}-qubit Simulation (Backend: {backend_choice}) ---")
        
        start_time = time.perf_counter()
        
        try:
            _public_api_logger.info(f"    - Initializing {num_qubits}-qubit lazy state...")
            state = create_quantum_state(num_qubits)
            
            _public_api_logger.info("    - Lazily building a complex circuit (Hadamards, Rzs, CNOTs)...")
            qc_complex = QuantumCircuit(num_qubits=num_qubits, description=f"Performance test circuit for {num_qubits} qubits")
            for i in range(num_qubits): qc_complex.h(i)
            for i in range(num_qubits): qc_complex.rz(i, math.pi / (i + 4))
            for i in range(num_qubits - 1): qc_complex.cnot(i, i + 1)
            
            _public_api_logger.info("    - Merging complex circuit (no computation yet)...")
            final_state_perf = run_circuit_on_state(state, qc_complex)
            assert not state._is_cache_valid, "Original state's cache must remain invalid (deep copy)."
            assert final_state_perf._is_cache_valid == False, "Final state's cache must be invalid until expanded."
            
            _public_api_logger.info("    - Triggering full state expansion and calculating probabilities...")
            _ = get_measurement_probabilities(final_state_perf) 
            assert final_state_perf._is_cache_valid, "Cache must be valid after expansion."
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            backend_in_use = final_state_perf._backend
            backend_name = type(backend_in_use).__name__
            
            _public_api_logger.info(f"    - Backend used by QuantumState: {backend_name}")
            _public_api_logger.info(f"    - Total simulation time (including expansion): {duration_ms:.2f} ms")
            
            assert duration_ms > 0, "Simulation duration should be positive"
            assert final_state_perf.is_valid(), "Final state of performance test should be valid."
            _public_api_logger.info("    - Final state is valid. Performance test successful.")

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _public_api_logger.critical(f"    - Performance test failed after {duration_ms:.2f} ms. Error: {e}", exc_info=True)
            raise e
        finally:
            _public_api_logger.info("--------------------------------------------------")
    
    
    
    # --- 步骤 4: 定义测试套件 ---
    
    # a) 将所有测试函数逻辑地分组
    core_functionality_tests = {
        "State and Circuit Creation (Lazy)": test_state_and_circuit_creation,
        "Gate Application & Circuit Run (Lazy Bell State)": test_gate_application_and_circuit_run,
        "Partial Trace & Von Neumann Entropy (Lazy/Optimized)": test_partial_trace_and_entropy,
        "Hamiltonian Expectation Value (Lazy)": test_hamiltonian_expectation,
        "Classical Control Flow (Lazy)": test_classical_control_flow,
        "New Gates (iSWAP, ECR)": test_new_gates_iswap_ecr,
        "Multi-Controlled Gates (MCZ, MCX, MCP, MCU)": test_multi_controlled_gates,
        # --- [新增的测试] ---
        "Parallel Computation Correctness (H-gate Kernel)": test_parallel_computation_correctness,
        "Kernel vs. Matrix Consistency": test_kernel_vs_matrix_consistency,
        # --- [新增结束] ---
    }

    advanced_algorithm_tests = {
        "QFT & IQFT Correctness (Lazy)": test_qft_and_inverse_qft,
        "QPE Circuit Builder": test_qpe_circuit_builder,
        "Hardware-Efficient Ansatz Builder": test_hardware_efficient_ansatz_builder,
    }

    noise_model_tests = {
        "Coherent Noise Model (Forces Mode Switch)": test_coherent_noise_model,
        "Hardware Noise Model (Forces Mode Switch)": test_hardware_noise_model,
        "Readout Noise Application (Classic)": test_readout_noise_application,
        "Crosstalk Noise Model (Unitary Replacement)": test_crosstalk_model,
    }
    
    # b) 组合所有单元测试
    all_unit_tests = {
        **core_functionality_tests,
        **advanced_algorithm_tests,
        **noise_model_tests,
    }

    # c) 创建最终的测试配置列表，为每个后端生成测试任务
    all_test_configs = []
    
    # 添加 PurePython 后端的测试
    for name, func in all_unit_tests.items():
        all_test_configs.append({'name': name, 'func': func, 'backend': 'pure_python'})
        
    # 动态添加 CuPy 后端的测试（如果可用）
    if cp is not None:
        for name, func in all_unit_tests.items():
            all_test_configs.append({'name': f"{name} (CuPy)", 'func': func, 'backend': 'cupy'})
    else:
        _public_api_logger.info("\n" + "#"*40)
        _public_api_logger.info("--- CuPy not found, skipping CuPy Backend Tests ---")
        _public_api_logger.info("#"*40 + "\n")

    # --- 步骤 5: 执行测试流程 (采用快速失败策略) ---
    
    results_summary = []
    all_tests_passed = True # 添加一个标志来跟踪整体状态

    # a) 首先，在单线程模式下运行所有单元测试
    _public_api_logger.info("\n" + "="*80)
    _public_api_logger.info("--- Running All Unit Tests (Fail-Fast Mode Enabled) ---")
    _public_api_logger.info("="*80 + "\n")
    
    for test_config in all_test_configs:
        success = run_test(test_config['name'], test_config['func'], test_config['backend'])
        results_summary.append({'name': test_config['name'], 'backend': test_config['backend'], 'success': success})
        
        # --- [核心修改] ---
        # 如果当前测试失败，立即中断测试套件
        if not success:
            all_tests_passed = False
            # 打印一个醒目的中断信息
            print("\n" + "!"*80)
            print("--- TEST SUITE HALTED: A critical error was encountered. ---")
            print(f"--- Failing Test: {test_config['name']} (Backend: {test_config['backend']}) ---")
            print("--- See logs above for the detailed error traceback. ---")
            print("!"*80 + "\n")
            break # 立即跳出 for 循环
            
    # b) 只有在所有单元测试都通过时，才继续运行性能演示
    if all_tests_passed:
        _public_api_logger.info("\n" + "="*80)
        _public_api_logger.info("--- All unit tests passed. Proceeding to Performance Demonstrations ---")
        _public_api_logger.info("="*80 + "\n")
        
        try:
            enable_parallelism() 
            if _parallel_enabled:
                # 运行 Pure Python 并行性能测试
                run_test(
                    "Performance Demo (Pure Python, Parallel)",
                    lambda: test_large_scale_simulation_performance(10, "pure_python"),
                    "pure_python"
                )

                if cp is not None:
                    run_test(
                        "Performance Demo (CuPy)",
                        lambda: test_large_scale_simulation_performance(15, "cupy"),
                        "cupy"
                    )
            else:
                _public_api_logger.info("\n⚠️ SKIPPED: Performance demonstrations were skipped because parallelism could not be enabled.")

        except Exception as e:
            _public_api_logger.critical(f"\n❌ An error occurred during the performance demonstration block: {type(e).__name__}: {e}", exc_info=True)
            all_tests_passed = False # 标记为失败
        finally:
            disable_parallelism()

    # --- 步骤 6: 打印最终总结 ---
    passed_count = sum(1 for r in results_summary if r['success'])
    total_run_count = len(results_summary) # 实际运行的测试数量
    
    _public_api_logger.info("\n" + "="*80)
    _public_api_logger.info("--- Overall Test Summary ---")
    
    for result in results_summary:
        status_icon = "✅ PASSED" if result['success'] else "❌ FAILED"
        _public_api_logger.info(f"{status_icon:<10} | {result['name']:<65} | Backend: {result['backend']}")
        
    _public_api_logger.info("-" * 80)
    
    if all_tests_passed:
        _public_api_logger.info(f"\033[92m🎉 All {total_run_count} unit tests and performance demos passed!\033[0m")
    else:
        _public_api_logger.critical(f"\033[91m🔥 Summary: {passed_count} / {total_run_count} tests passed before halting.\033[0m")
    
    _public_api_logger.info("="*80 + "\n")

    if not all_tests_passed:
        _public_api_logger.critical("Some tests failed. Exiting with status code 1.")
        sys.exit(1)
    else:
        _public_api_logger.info("All tests passed successfully. Exiting with status code 0.")
        sys.exit(0)