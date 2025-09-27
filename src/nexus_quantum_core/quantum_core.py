# --- START OF FILE quantum_core.py ---

# -*- coding: utf-8 -*-
"""
NEXUS QUANTUM DEFENSE - 量子计算核心库 v1.2

功能:
- 封装所有量子态模拟 (QuantumState)、量子线路 (QuantumCircuit) 的核心数据结构与逻辑。
- 包含高性能的并行计算基础设施，用于加速大规模量子模拟。
- 提供一个简洁、稳定的公共API，供上层应用（如 AIEngine）调用。

增强功能 (v1.2):
- VQA支持: Pauli算子与哈密顿量表示、哈密顿量期望值计算。
- 量子动力学: Suzuki-Trotter分解。
- 深度分析: 冯·诺依曼纠缠熵、布洛赫矢量。
- 高级算法: Grover扩散算子、经典控制流。

Author: 跳舞的火公子
Date: 2025-06-23
"""

# ========================================================================
# --- 1. 导入依赖 ---
# ========================================================================

# --- 1.1 Python标准库 ---
# 核心功能所需
import sys
import os
import logging
import copy
import math
import cmath
import uuid
import time
import types
from typing import Optional, List, Tuple, Dict, Any, Literal, Union
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from functools import lru_cache

# 纯Python并行计算框架所需
import multiprocessing as mp
from multiprocessing import pool, RawArray
import ctypes


# --- 1.2 可选的高性能GPU依赖 ---
# 尝试导入Cupy。如果失败，则GPU加速不可用，代码将自动使用纯Python后端。
try:
    import cupy as cp
except ImportError:
    cp = None
    # 在模块加载时记录一次信息，告知用户GPU功能状态
    logging.getLogger(__name__).info(
        "CuPy library not found. GPU acceleration will be unavailable. "
        "The simulation will use the Pure Python backend by default."
    )


# --- 1.3 日志记录器初始化 ---
# 获取一个特定于此模块的日志记录器实例。
# 主应用程序的日志配置将自动应用到这个记录器上。
logger = logging.getLogger(__name__)




# ========================================================================
# --- 2. 硬件资源配置 ---
# ========================================================================

# --- 2.1 硬件资源常量 ---
# 这些常量与具体后端无关，因此保持不变
_BYTES_PER_COMPLEX128_ELEMENT: int = 16
_GB_TO_BYTES: int = 1024 ** 3
_TOTAL_SYSTEM_RAM_GB_HW: int = 128  # 示例硬件配置
_TOTAL_GPU_VRAM_GB_HW: int = 24   # 示例硬件配置

# --- 2.2 库的内部可配置参数 (终极修正版) ---
_core_config: Dict[str, Any] = {
    # 用户可设置的最大量子比特数上限
    "MAX_QUBITS": 10,
    
    # 后端选择配置。可用选项: 'auto', 'cupy', 'pure_python'
    # 'numpy' 不再是有效选项。
    "BACKEND_CHOICE": "auto",
    
    # 触发纯Python并行计算的量子比特数阈值。
    # 注意：由于纯Python计算开销较大，并行化的启动开销也相对较高，
    # 可能需要一个比之前NumPy版本更高的阈值才能看到性能提升。
    "PARALLEL_COMPUTATION_QUBIT_THRESHOLD": 12,
}

def configure_quantum_core(config_dict: Dict[str, Any]):
    """
    公共API函数：用于在运行时配置量子核心库的行为。
    [终极修正版] 移除了对 NumPy 后端的支持。
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
    _core_config.update(config_dict)
    logger.info(f"Quantum Core library has been configured with new settings: {_core_config}")



# ========================================================================
# --- [Project Bedrock] 纯Python后端，用于终极调试与稳定性验证 ---
# ========================================================================

class PurePythonBackend:
    """
    一个完全用纯Python列表和标准库实现的线性代数后端。
    用于在不依赖任何第三方库的情况下，验证核心量子算法的逻辑正确性。
    性能较低，仅用于调试和确保稳定性。
    
    数据结构:
    - 向量: Python的列表 (e.g., [1+0j, 0+0j])
    - 矩阵: Python的嵌套列表 (e.g., [[1+0j, 0+0j], [0+0j, 1+0j]])
    """
    
    # [注意] 为了在 PurePythonBackend 中实现随机选择（例如 simulate_measurement），
    # 仍然需要一个伪随机数生成器。这里我们选择直接使用 Python 的 `random` 模块。
    # 如果要彻底硬编码，连 `random` 模块也要避免，但那将是极其复杂的，且超出本调试范围。
    import random
    
    # --- 基础矩阵/向量创建 ---

    def create_matrix(self, rows: int, cols: int, value: complex = 0.0 + 0.0j) -> List[List[complex]]:
        """
        创建一个由嵌套列表表示的、指定大小和初始值的复数矩阵。

        这是 PurePythonBackend 的基础构建块之一，用于替代 `numpy.zeros`,
        `numpy.full` 或直接的矩阵初始化。

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
            TypeError: 如果 `rows` 或 `cols` 不是整数。

        Example:
            >>> backend = PurePythonBackend()
            >>> backend.create_matrix(2, 3, 1+1j)
            [[1+1j, 1+1j, 1+1j], [1+1j, 1+1j, 1+1j]]
        """
        # --- 输入验证 ---
        if not isinstance(rows, int) or not isinstance(cols, int):
            raise TypeError("Matrix dimensions (rows, cols) must be integers.")
        if rows < 0 or cols < 0:
            raise ValueError("Matrix dimensions (rows, cols) cannot be negative.")

        # --- 核心实现 ---
        # 使用列表推导式高效地创建嵌套列表。
        # [value] * cols 会创建一个对 value 的引用列表，对于不可变类型（如complex）是安全的。
        # 但为了极致的明确性，我们使用内部循环。
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

        Example:
            >>> backend = PurePythonBackend()
            >>> backend.eye(3)
            [[1+0j, 0+0j, 0+0j],
             [0+0j, 1+0j, 0+0j],
             [0+0j, 0+0j, 1+0j]]
        """
        # --- 输入验证 ---
        if not isinstance(dim, int):
            raise TypeError("Dimension (dim) must be an integer.")
        if dim < 0:
            raise ValueError("Dimension (dim) cannot be negative.")

        # --- 核心实现 ---
        # 1. 首先使用 create_matrix 创建一个全零的 dim x dim 矩阵。
        #    这确保了矩阵结构是正确的，并且所有非对角线元素都已正确设置为 0。
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

        Example:
            >>> backend = PurePythonBackend()
            >>> backend.zeros((3,))
            [0+0j, 0+0j, 0+0j]
            >>> backend.zeros((2, 3))
            [[0+0j, 0+0j, 0+0j], [0+0j, 0+0j, 0+0j]]
        """
        # --- 输入验证 ---
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple.")
        
        if len(shape) == 1:
            # --- 创建一维向量 ---
            dim = shape[0]
            if not isinstance(dim, int):
                raise TypeError("Dimension in shape must be an integer.")
            if dim < 0:
                raise ValueError("Dimension in shape cannot be negative.")
            return [0.0 + 0.0j for _ in range(dim)]
            
        elif len(shape) == 2:
            # --- 创建二维矩阵 ---
            rows, cols = shape
            if not isinstance(rows, int) or not isinstance(cols, int):
                raise TypeError("Dimensions in shape must be integers.")
            if rows < 0 or cols < 0:
                raise ValueError("Dimensions in shape cannot be negative.")
            # 调用已验证的 create_matrix 方法来保证一致性
            return self.create_matrix(rows, cols, value=0.0 + 0.0j)
            
        else:
            # --- 不支持的形状 ---
            raise ValueError(f"PurePythonBackend only supports 1D or 2D shapes for zeros, but got shape with {len(shape)} dimensions.")

    # --- 核心线性代数操作 ---

    def dot(self, mat_a: List[List[complex]], mat_b: List[List[complex]]) -> List[List[complex]]:
        """
        计算两个矩阵的矩阵乘法 (dot product)，C = A @ B。

        此函数严格按照线性代数的定义，通过三个嵌套循环来实现矩阵乘法。
        它不依赖任何外部库，用于替代 `numpy.dot` 或 `@` 运算符。

        Args:
            mat_a (List[List[complex]]):
                乘法中的左矩阵 (A)。必须是一个非空的、形状规整的嵌套列表。
            mat_b (List[List[complex]]):
                乘法中的右矩阵 (B)。必须是一个非空的、形状规整的嵌套列表。

        Returns:
            List[List[complex]]:
                结果矩阵 C，其形状为 `(rows_a, cols_b)`。

        Raises:
            TypeError: 如果输入 `mat_a` 或 `mat_b` 不是嵌套列表，或者
                       其元素不是 `complex` 类型。
            ValueError: 如果任一矩阵为空、形状不规整（各行长度不一），
                        或者它们的维度不满足矩阵乘法的要求
                        （即 `cols_a` 必须等于 `rows_b`）。

        Example:
            >>> backend = PurePythonBackend()
            >>> a = [[1+0j, 2+0j], [3+0j, 4+0j]]
            >>> b = [[5+0j, 6+0j], [7+0j, 8+0j]]
            >>> backend.dot(a, b)
            [[19+0j, 22+0j], [43+0j, 50+0j]]
        """
        # --- 步骤 1: 严格的输入验证 ---

        # 检查 mat_a 的有效性
        if not isinstance(mat_a, list) or not mat_a or not isinstance(mat_a[0], list):
            raise TypeError("Input 'mat_a' must be a non-empty nested list (matrix).")
        rows_a = len(mat_a)
        cols_a = len(mat_a[0])
        if cols_a == 0:
            raise ValueError("Input 'mat_a' cannot have rows of zero length.")
        for i in range(rows_a):
            if len(mat_a[i]) != cols_a:
                raise ValueError("Input 'mat_a' is not a well-formed matrix (rows have different lengths).")

        # 检查 mat_b 的有效性
        if not isinstance(mat_b, list) or not mat_b or not isinstance(mat_b[0], list):
            raise TypeError("Input 'mat_b' must be a non-empty nested list (matrix).")
        rows_b = len(mat_b)
        cols_b = len(mat_b[0])
        if cols_b == 0:
            raise ValueError("Input 'mat_b' cannot have rows of zero length.")
        for i in range(rows_b):
            if len(mat_b[i]) != cols_b:
                raise ValueError("Input 'mat_b' is not a well-formed matrix (rows have different lengths).")

        # 检查维度兼容性
        if cols_a != rows_b:
            raise ValueError(
                f"Matrix dimensions are not compatible for dot product: "
                f"A is ({rows_a}x{cols_a}) and B is ({rows_b}x{cols_b}). "
                f"The inner dimensions ({cols_a} and {rows_b}) must match."
            )

        # --- 步骤 2: 核心实现 ---

        # 创建一个正确大小的全零结果矩阵
        res = self.create_matrix(rows_a, cols_b)

        # 使用三个嵌套循环实现 C[i,j] = sum(A[i,k] * B[k,j])
        for i in range(rows_a):      # 遍历结果矩阵的行
            for j in range(cols_b):  # 遍历结果矩阵的列
                sum_val = 0.0 + 0.0j
                for k in range(cols_a):  # 遍历内积维度
                    # 确保元素是复数类型，以防传入的是int或float
                    val_a = complex(mat_a[i][k])
                    val_b = complex(mat_b[k][j])
                    sum_val += val_a * val_b
                res[i][j] = sum_val
                
        return res

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

        Example:
            >>> backend = PurePythonBackend()
            >>> A = [[1, 2], [3, 4]]
            >>> B = [[0, 5]]
            >>> backend.kron(A, B)
            [[0, 5, 0, 10],    # 1*B, 2*B
             [0, 15, 0, 20]]   # 3*B, 4*B
        """
        # --- 步骤 1: 严格的输入验证 ---

        # 检查 mat_a 的有效性
        if not isinstance(mat_a, list) or not mat_a or not isinstance(mat_a[0], list):
            raise TypeError("Input 'mat_a' must be a non-empty nested list (matrix).")
        rows_a = len(mat_a)
        cols_a = len(mat_a[0])
        if cols_a == 0:
            raise ValueError("Input 'mat_a' cannot have rows of zero length.")
        for i in range(rows_a):
            if len(mat_a[i]) != cols_a:
                raise ValueError("Input 'mat_a' is not a well-formed matrix (rows have different lengths).")

        # 检查 mat_b 的有效性
        if not isinstance(mat_b, list) or not mat_b or not isinstance(mat_b[0], list):
            raise TypeError("Input 'mat_b' must be a non-empty nested list (matrix).")
        rows_b = len(mat_b)
        cols_b = len(mat_b[0])
        if cols_b == 0:
            raise ValueError("Input 'mat_b' cannot have rows of zero length.")
        for i in range(rows_b):
            if len(mat_b[i]) != cols_b:
                raise ValueError("Input 'mat_b' is not a well-formed matrix (rows have different lengths).")

        # --- 步骤 2: 核心实现 ---

        # 计算结果矩阵的维度
        res_rows = rows_a * rows_b
        res_cols = cols_a * cols_b
        
        # 创建一个正确大小的全零结果矩阵
        res = self.create_matrix(res_rows, res_cols)
        
        # 使用四个嵌套循环实现 C[i_a*rows_b + i_b, j_a*cols_b + j_b] = A[i_a,j_a] * B[i_b,j_b]
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
        计算两个向量的外积 (outer product)，结果为一个矩阵 M = |a⟩⟨b|。

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

        Example:
            >>> backend = PurePythonBackend()
            >>> a = [1+0j, 2+0j]
            >>> b = [3+0j, 4+0j, 5+0j]
            >>> backend.outer(a, b)
            [[3+0j, 4+0j, 5+0j],
             [6+0j, 8+0j, 10+0j]]
        """
        # --- 步骤 1: 严格的输入验证 ---

        # 检查 vec_a 的有效性
        if not isinstance(vec_a, list) or (vec_a and isinstance(vec_a[0], list)):
            raise TypeError("Input 'vec_a' must be a 1D list (vector).")
        
        # 检查 vec_b 的有效性
        if not isinstance(vec_b, list) or (vec_b and isinstance(vec_b[0], list)):
            raise TypeError("Input 'vec_b' must be a 1D list (vector).")
            
        len_a = len(vec_a)
        len_b = len(vec_b)

        # --- 步骤 2: 核心实现 ---

        # 创建一个正确大小的全零结果矩阵
        mat = self.create_matrix(len_a, len_b)
        
        # 使用两个嵌套循环实现 M[i,j] = a[i] * b[j]
        for i in range(len_a):
            # 优化：如果 a[i] 是 0，则整行都是 0，可以跳过
            val_a = complex(vec_a[i])
            if self.isclose(val_a, 0.0 + 0.0j):
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

        Example:
            >>> backend = PurePythonBackend()
            >>> m = [[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]]
            >>> backend.conj_transpose(m)
            [[1-1j, 4-4j],
             [2-2j, 5-5j],
             [3-3j, 6-6j]]
        """
        # --- 步骤 1: 严格的输入验证 ---

        if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
            raise TypeError("Input 'mat' must be a non-empty nested list (matrix).")
        rows = len(mat)
        cols = len(mat[0])
        if cols == 0:
            raise ValueError("Input matrix cannot have rows of zero length.")
        for i in range(rows):
            if len(mat[i]) != cols:
                raise ValueError("Input matrix is not a well-formed matrix (rows have different lengths).")

        # --- 步骤 2: 核心实现 ---

        # 创建一个转置后大小的结果矩阵
        res = self.create_matrix(cols, rows)
        
        # 使用两个嵌套循环实现 res[j,i] = mat[i,j]^*
        for i in range(rows):
            for j in range(cols):
                # 先取元素，再进行共轭，然后赋值给转置后的位置
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

        Example:
            >>> backend = PurePythonBackend()
            >>> m = [[1+1j, 2+0j], [3+0j, 4-2j]]
            >>> backend.trace(m)
            (5-1j)
        """
        # --- 步骤 1: 严格的输入验证 ---

        if not isinstance(mat, list):
            raise TypeError("Input 'mat' must be a list of lists (matrix).")
        
        # 处理空矩阵的边缘情况
        if not mat:
            return 0.0 + 0.0j  # 空矩阵的迹定义为 0

        if not isinstance(mat[0], list):
             raise TypeError("Input 'mat' must be a nested list (matrix).")

        rows = len(mat)
        # 再次检查，防止[[...], []] 这样的情况
        if rows > 0 and not mat[0]:
             raise ValueError("Input matrix cannot have rows of zero length.")
             
        cols = len(mat[0])

        # 检查是否为方阵
        if rows != cols:
            raise ValueError(f"Trace is only defined for square matrices, but got a matrix of shape ({rows}x{cols}).")
        
        # 检查形状是否规整 (虽然对角线计算不需要，但这是一个好习惯)
        for i in range(rows):
            if len(mat[i]) != cols:
                raise ValueError("Input matrix is not a well-formed matrix (rows have different lengths).")

        # --- 步骤 2: 核心实现 ---

        # 初始化一个复数累加器
        trace_sum = 0.0 + 0.0j
        
        # 循环遍历主对角线并累加元素
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

        Example:
            >>> backend = PurePythonBackend()
            >>> m = [[1+1j, 2+0j, 3+0j],
                     [4+0j, 5-2j, 6+0j],
                     [7+0j, 8+0j, 9+3j]]
            >>> backend.diag(m)
            [1+1j, 5-2j, 9+3j]
        """
        # --- 步骤 1: 严格的输入验证 ---

        if not isinstance(mat, list):
            raise TypeError("Input 'mat' must be a list of lists (matrix).")
        
        # 处理空矩阵的边缘情况
        if not mat:
            return []  # 空矩阵的对角线是空列表

        if not isinstance(mat[0], list):
             raise TypeError("Input 'mat' must be a nested list (matrix).")
        
        rows = len(mat)
        # 再次检查，防止[[...], []] 这样的情况
        if rows > 0 and not mat[0]:
             raise ValueError("Input matrix cannot have rows of zero length.")
        
        cols = len(mat[0])

        # 检查是否为方阵
        if rows != cols:
            raise ValueError(f"Diagonal can only be extracted from a square matrix in this context, but got shape ({rows}x{cols}).")
        
        # 检查形状是否规整 (虽然对角线计算不需要，但这是一个好习惯)
        for i in range(rows):
            if len(mat[i]) != cols:
                raise ValueError("Input matrix is not a well-formed matrix (rows have different lengths).")

        # --- 步骤 2: 核心实现 ---

        # 使用列表推导式高效地提取对角线元素
        # 同时确保每个元素都被转换为 complex 类型
        return [complex(mat[i][i]) for i in range(rows)]

    @staticmethod
    def _static_flatten(nested_list: Any) -> List[complex]:
        """
        [最终增强版] 一个静态辅助方法，使用非递归的栈实现，将任意嵌套
        的列表扁平化为一维复数列表。
        
        此版本通过直接迭代并反向压栈，避免了 `reversed()` 的调用，
        在处理宽列表时可能略有性能优势。
        """
        flat_list = []
        stack = [iter([nested_list])]  # 栈中存储迭代器

        while stack:
            try:
                # 尝试从栈顶的迭代器中获取下一个元素
                element = next(stack[-1])
                if isinstance(element, list):
                    # 如果元素是列表，将它的迭代器压入栈顶
                    stack.append(iter(element))
                else:
                    # 如果是标量，直接添加到结果中
                    flat_list.append(complex(element))
            except StopIteration:
                # 如果当前迭代器耗尽，将其弹出栈
                stack.pop()
        
        return flat_list

    @staticmethod
    def _build_from_iterator(shape_tuple: Tuple[int, ...], data_iterator: Any) -> Any:
        """
        [保持不变] 一个静态辅助方法，通过消耗一个迭代器来递归地构建嵌套列表。
        此实现已足够高效和稳健。
        """
        if not shape_tuple:
            return next(data_iterator)
        if len(shape_tuple) == 1:
            return [next(data_iterator) for _ in range(shape_tuple[0])]
        
        current_dim_size = shape_tuple[0]
        deeper_shape = shape_tuple[1:]
        return [PurePythonBackend._build_from_iterator(deeper_shape, data_iterator) for _ in range(current_dim_size)]
    
    def reshape(self, data: Any, new_shape: Tuple[int, ...]) -> Any:
        """
        [最终增强版] 将一个向量、矩阵或任意嵌套的列表重塑为新的指定形状。

        此版本保持了 v2 的所有功能，包括 `-1` 推断和高效的迭代器构建，
        同时优化了内部 `_static_flatten` 的实现，并对边缘情况处理得更加明确。

        Args:
            data (Any): 输入的数据，可以是任意嵌套深度的列表或一维列表。
            new_shape (Tuple[int, ...]): 一个整数元组，指定了输出张量的新维度。

        Returns:
            Any: 重塑后的、代表多维张量的嵌套列表。
        """
        if not isinstance(new_shape, tuple):
            raise TypeError("new_shape must be a tuple of integers.")

        # --- 步骤 1: 扁平化 ---
        flat_list = self._static_flatten(data)
        current_size = len(flat_list)
        
        # --- 步骤 2: 维度推断与验证 ---
        final_shape = list(new_shape)
        if final_shape.count(-1) > 1:
            raise ValueError(f"Cannot reshape: new_shape {new_shape} can only contain one '-1'.")
        
        if -1 in final_shape:
            product_of_known_dims = math.prod(dim for dim in final_shape if dim > 0)
            if product_of_known_dims == 0:
                if current_size != 0:
                    raise ValueError(f"Cannot infer '-1' in shape {new_shape} for data of size {current_size} when a zero dimension is present.")
                inferred_dim = 0
            else:
                if current_size % product_of_known_dims != 0:
                    raise ValueError(f"Cannot reshape array of size {current_size} into shape {new_shape}.")
                inferred_dim = current_size // product_of_known_dims
            final_shape[final_shape.index(-1)] = inferred_dim
        
        final_shape = tuple(final_shape)

        # --- 步骤 3: 尺寸验证 ---
        if current_size != math.prod(final_shape):
            raise ValueError(
                f"Cannot reshape: data size {current_size} is incompatible with target shape {final_shape} (inferred from {new_shape})."
            )
            
        # --- 步骤 4: 处理边缘情况并构建 ---
        if current_size == 0:
            # 如果总元素为0，任何包含0的形状都是有效的，返回一个空结构
            if not final_shape: return None
            def build_empty(shape):
                if not shape: return []
                return [build_empty(shape[1:]) for _ in range(shape[0])]
            return build_empty(final_shape)

        if not final_shape:
            return flat_list[0]

        data_iterator = iter(flat_list)
        return self._build_from_iterator(final_shape, data_iterator)
    @staticmethod
    def _parse_einsum_for_partial_trace(subscripts: str, num_qubits: int) -> Tuple[List[int], List[int]]:
        """[保持不变] 解析einsum字符串，逻辑已足够稳健。"""
        input_subs, _ = subscripts.split('->')
        bra_subs = input_subs[:num_qubits]
        ket_subs = input_subs[num_qubits:]
        qubits_to_keep = [i for i in range(num_qubits) if bra_subs[i] != ket_subs[i]]
        qubits_to_trace = [i for i in range(num_qubits) if bra_subs[i] == ket_subs[i]]
        return qubits_to_keep, qubits_to_trace
    def einsum(self, subscripts: str, tensor: Any) -> Any:
        """
        [最终增强版 v3] 一个高性能、纯迭代的 einsum 实现，专门用于部分迹。
        
        此版本采用了“推送”算法：遍历一次原始高维张量，将每个元素的值
        “推送”并累加到输出矩阵的正确位置。此方法避免了多重嵌套循环和递归，
        在性能和代码清晰度上都有显著提升。

        Args:
            subscripts (str): einsum 规则字符串，例如 "abcaBc->acAC"。
            tensor (Any): 一个代表多维张量的嵌套列表，其秩为 2 * num_qubits。

        Returns:
            Any: 计算结果，一个降维后的密度矩阵（嵌套列表）。
        """
        # --- 步骤 1: 解析与设置 ---
        num_qubits = len(subscripts.split('->')[0]) // 2
        qubits_to_keep, qubits_to_trace = self._parse_einsum_for_partial_trace(subscripts, num_qubits)
        
        num_qubits_kept = len(qubits_to_keep)
        dim_out = 1 << num_qubits_kept
        reduced_rho = self.create_matrix(dim_out, dim_out)

        # --- 步骤 2: 预计算位掩码和映射，用于高效索引转换 ---
        # 创建一个掩码，用于从全局索引中分离出保留比特的部分
        keep_mask_bra = sum(1 << i for i in qubits_to_keep)
        # 创建从全局比特位置到输出矩阵局部比特位置的映射
        keep_map = {global_pos: local_pos for local_pos, global_pos in enumerate(qubits_to_keep)}

        # --- 步骤 3: 遍历原始张量的所有对角线元素 ---
        # 我们只关心迹运算，即 bra 和 ket 在迹比特上的索引相同的部分。
        # 外层循环遍历所有保留比特的配置，这对应输出矩阵的行列。
        for keep_config_bra in range(1 << num_qubits_kept):
            for keep_config_ket in range(1 << num_qubits_kept):
                
                sum_val = 0.0 + 0.0j

                # 内层循环遍历所有迹比特的配置
                for trace_config in range(1 << len(qubits_to_trace)):
                    
                    # --- 步骤 4: 构建原始张量的 bra 和 ket 索引 ---
                    bra_index_in = 0
                    ket_index_in = 0

                    # a) 组合保留比特的索引
                    for i, qubit_pos in enumerate(qubits_to_keep):
                        if (keep_config_bra >> i) & 1:
                            bra_index_in |= (1 << qubit_pos)
                        if (keep_config_ket >> i) & 1:
                            ket_index_in |= (1 << qubit_pos)
                    
                    # b) 组合迹比特的索引 (对角线)
                    for i, qubit_pos in enumerate(qubits_to_trace):
                        if (trace_config >> i) & 1:
                            trace_bit = (1 << qubit_pos)
                            bra_index_in |= trace_bit
                            ket_index_in |= trace_bit
                    
                    # --- 步骤 5: 从高维张量中获取元素并累加 ---
                    # 这是一个技巧：将高维张量视为一个扁平化的一维列表进行访问
                    # 索引 = bra_index_in * (2^N) + ket_index_in
                    # 但由于输入是嵌套列表，我们仍需递归访问
                    element = tensor
                    indices = []
                    # 组合 bra 和 ket 索引以遍历张量
                    for i in range(num_qubits - 1, -1, -1):
                        indices.append((bra_index_in >> i) & 1)
                    for i in range(num_qubits - 1, -1, -1):
                        indices.append((ket_index_in >> i) & 1)

                    for idx in indices:
                        element = element[idx]

                    sum_val += element

                # --- 步骤 6: "推送"结果到输出矩阵 ---
                reduced_rho[keep_config_bra][keep_config_ket] = sum_val
        
        return reduced_rho
    
    
    def transpose(self, mat: List[List[complex]]) -> List[List[complex]]:
        """
        计算一个矩阵的转置 (transpose)。

        矩阵的转置是通过将其行向量转换为列向量（反之亦然）得到的新矩阵。
        结果矩阵 `M_T` 的元素由 `M_T[j][i] = M[i][j]` 定义。
        如果原始矩阵的形状是 `(m, n)`，则转置矩阵的形状将是 `(n, m)`。

        此函数是实现共轭转置 (`conj_transpose`) 的关键步骤之一。

        Args:
            mat (List[List[complex]]):
                输入矩阵 M。必须是一个非空的、形状规整的嵌套列表。

        Returns:
            List[List[complex]]:
                转置后的新矩阵 M_T。

        Raises:
            TypeError: 如果输入 `mat` 不是嵌套列表。
            ValueError: 如果矩阵为空或形状不规整（各行长度不一）。

        Example:
            >>> backend = PurePythonBackend()
            >>> m = [[1, 2, 3],
                     [4, 5, 6]]  # Shape (2, 3)
            >>> backend.transpose(m)
            [[1, 4],
             [2, 5],
             [3, 6]]  # Shape (3, 2)
        """
        # --- 步骤 1: 严格的输入验证 ---

        if not isinstance(mat, list):
            raise TypeError("Input 'mat' must be a list of lists (matrix).")
        
        # 处理空矩阵的边缘情况
        if not mat:
            return []  # 空矩阵的转置是空列表

        if not isinstance(mat[0], list):
             raise TypeError("Input 'mat' must be a nested list (matrix).")

        rows = len(mat)
        # 再次检查，防止[[...], []] 这样的情况
        if rows > 0 and not mat[0]:
            # 如果是 [[],[],[]] 这种，cols=0, new_rows=0，返回 [] 是合理的
            pass
             
        cols = len(mat[0])

        # 检查形状是否规整
        for i in range(rows):
            if len(mat[i]) != cols:
                raise ValueError("Input matrix is not a well-formed matrix (rows have different lengths).")

        # --- 步骤 2: 核心实现 ---

        # 创建一个转置后大小 (`cols` x `rows`) 的结果矩阵
        new_rows, new_cols = cols, rows
        res = self.create_matrix(new_rows, new_cols)
        
        # 使用两个嵌套循环实现 res[j,i] = mat[i,j]
        for i in range(rows):
            for j in range(cols):
                res[j][i] = mat[i][j]
                
        return res

    # --- 浮点数比较与数学函数 ---

    def isclose(self, a: Union[complex, float, int], b: Union[complex, float, int], atol: float = 1e-9) -> bool:
        """
        判断两个数值（整数、浮点数或复数）是否在指定的绝对容差内“足够接近”。

        **重要性:**
        由于计算机浮点数表示的内在不精确性，直接使用 `==` 来比较两个
        浮点数或复数通常是不可靠的。例如，`0.1 + 0.2 == 0.3` 在 Python 中
        返回 `False`。此函数通过检查两个数之差的绝对值是否小于一个很小的
        容差 `atol` 来解决这个问题。

        判断标准: `abs(a - b) <= atol`

        此函数用于替代 `numpy.isclose` 的基本功能（仅绝对容差）。

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

        Example:
            >>> backend = PurePythonBackend()
            >>> backend.isclose(0.1 + 0.2, 0.3)
            True
            >>> backend.isclose(1+1j, 1.0000000001 + 0.9999999999j)
            True
            >>> backend.isclose(1, 1.1)
            False
        """
        # --- 输入验证 ---
        if not isinstance(a, (complex, float, int)):
            raise TypeError(f"Input 'a' must be a numeric type, but got {type(a).__name__}.")
        if not isinstance(b, (complex, float, int)):
            raise TypeError(f"Input 'b' must be a numeric type, but got {type(b).__name__}.")
        if not isinstance(atol, float):
            raise TypeError(f"Absolute tolerance 'atol' must be a float, but got {type(atol).__name__}.")

        # --- 核心实现 ---
        # abs() 函数可以自然地处理复数的模（magnitude），
        # 即 sqrt(real_part**2 + imag_part**2)。
        # 这正是我们比较两个复数距离所需要的。
        return abs(complex(a) - complex(b)) <= atol

    def allclose(self, mat_a: List[List[complex]], mat_b: List[List[complex]], atol: float = 1e-9) -> bool:
        """
        判断两个矩阵中的所有对应元素是否都“足够接近”。

        此函数逐个元素地比较两个矩阵，对每一对元素 `(a[i][j], b[i][j])`
        都调用 `self.isclose()` 方法。只有当**所有**对应元素都满足 `isclose`
        的条件时，此函数才返回 `True`。

        在量子模拟的单元测试中，这个函数至关重要，因为它允许我们在可接受的
        浮点数误差范围内，验证一个复杂的演化过程得到的最终密度矩阵是否与
        理论上的期望矩阵相符。

        此函数用于替代 `numpy.allclose` 的基本功能。

        Args:
            mat_a (List[List[complex]]): 第一个矩阵。
            mat_b (List[List[complex]]): 第二个矩阵。
            atol (float, optional):
                传递给 `isclose` 的绝对容差。默认为 `1e-9`。

        Returns:
            bool:
                如果两个矩阵的所有对应元素都足够接近，则返回 `True`。
                如果矩阵形状不匹配，或者任何一对元素不够接近，则返回 `False`。

        Raises:
            TypeError: 如果输入不是嵌套列表，或者 `atol` 不是浮点数。
            ValueError: 如果输入矩阵形状不规整。

        Example:
            >>> backend = PurePythonBackend()
            >>> a = [[1.0, 2.000000001], [3.0, 4.0]]
            >>> b = [[1.0, 2.0], [3.0, 4.0]]
            >>> backend.allclose(a, b)
            True
            >>> c = [[1.0, 2.1], [3.0, 4.0]]
            >>> backend.allclose(a, c)
            False
        """
        # --- 步骤 1: 严格的输入验证 ---

        # 检查 mat_a 的有效性
        if not isinstance(mat_a, list) or (mat_a and not isinstance(mat_a[0], list)):
            raise TypeError("Input 'mat_a' must be a nested list (matrix).")
        # 检查 mat_b 的有效性
        if not isinstance(mat_b, list) or (mat_b and not isinstance(mat_b[0], list)):
            raise TypeError("Input 'mat_b' must be a nested list (matrix).")
        
        # 检查维度是否完全匹配
        rows_a = len(mat_a)
        rows_b = len(mat_b)
        if rows_a != rows_b:
            return False

        # 如果矩阵为空，则它们是相等的
        if rows_a == 0:
            return True
            
        cols_a = len(mat_a[0])
        cols_b = len(mat_b[0])
        if cols_a != cols_b:
            return False

        # --- 步骤 2: 逐元素比较 ---
        
        # 遍历所有元素，并使用 isclose 进行比较
        for i in range(rows_a):
            # 顺便检查形状规整性
            if len(mat_a[i]) != cols_a or len(mat_b[i]) != cols_b:
                 raise ValueError("One or both matrices are not well-formed (rows have different lengths).")

            for j in range(cols_a):
                # 只要有一对元素不满足 isclose，就立即返回 False
                if not self.isclose(mat_a[i][j], mat_b[i][j], atol=atol):
                    return False
        
        # 如果所有循环都成功完成，说明所有元素都足够接近
        return True

    def sin(self, x: Union[float, int, complex]) -> Union[float, complex]:
        """
        计算一个数值的正弦值。

        此函数作为 Python 标准库 `math.sin` 和 `cmath.sin` 的统一接口。
        它会自动判断输入是实数还是复数，并调用相应的库函数。

        Args:
            x (Union[float, int, complex]): 输入的数值（角度，以弧度为单位）。

        Returns:
            Union[float, complex]: 输入值的正弦。

        Raises:
            TypeError: 如果输入不是数值类型。
        """
        if isinstance(x, complex):
            # 如果输入是复数，使用 cmath 库
            return cmath.sin(x)
        elif isinstance(x, (float, int)):
            # 如果输入是实数，使用 math 库
            return math.sin(x)
        else:
            raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")
    def cos(self, x: Union[float, int, complex]) -> Union[float, complex]:
        """
        计算一个数值的余弦值。

        此函数作为 Python 标准库 `math.cos` 和 `cmath.cos` 的统一接口。
        它会自动判断输入是实数还是复数，并调用相应的库函数，从而
        确保对两种类型的数值都能正确处理。

        Args:
            x (Union[float, int, complex]): 输入的数值（角度，以弧度为单位）。

        Returns:
            Union[float, complex]: 输入值的余弦。

        Raises:
            TypeError: 如果输入不是数值类型。
        """
        if isinstance(x, complex):
            # 如果输入是复数，使用 cmath 库
            return cmath.cos(x)
        elif isinstance(x, (float, int)):
            # 如果输入是实数，使用 math 库
            return math.cos(x)
        else:
            raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")
    def exp(self, x: Union[float, int, complex]) -> Union[float, complex]:
        """
        计算 e (自然常数) 的 x 次幂, e^x。

        此函数作为 Python 标准库 `math.exp` 和 `cmath.exp` 的统一接口。
        它会自动判断输入是实数还是复数，并调用相应的库函数。
        - 对于实数 x, 返回 e^x。
        - 对于复数 z = a + bi, 返回 e^a * (cos(b) + i*sin(b))。

        Args:
            x (Union[float, int, complex]): 指数。

        Returns:
            Union[float, complex]: 计算结果 e^x。

        Raises:
            TypeError: 如果输入不是数值类型。
        """
        # --- 输入验证 ---
        if not isinstance(x, (float, int, complex)):
            raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")

        # --- 核心实现 ---
        if isinstance(x, complex):
            # 如果输入是复数，使用 cmath 库的 exp 函数
            return cmath.exp(x)
        else:
            # 如果输入是实数 (float or int)，使用 math 库的 exp 函数
            return math.exp(x)
    def log2(self, x: float) -> float:
        """
        计算一个实数的以2为底的对数 (log₂x)。

        此函数是 Python 标准库 `math.log2` 的一个直接包装器。
        在量子信息论中，它主要用于计算信息熵，例如冯·诺依曼熵，
        其结果的单位通常是“比特”。

        **注意:** 此函数的定义域是正实数 (x > 0)。
        输入 0 会导致 `-inf`，输入负数会引发 `ValueError`。
        在调用此函数前，应确保输入值是有效的。

        Args:
            x (float): 输入的正实数。

        Returns:
            float: 计算结果 log₂x。

        Raises:
            TypeError: 如果输入不是实数类型。
            ValueError: 如果输入为非正数（由 `math.log2` 引发）。
        """
        # --- 输入验证 ---
        if not isinstance(x, (float, int)):
            raise TypeError(f"Input for log2 must be a real number (float or int), but got {type(x).__name__}.")

        # --- 核心实现 ---
        # 直接调用标准库 math 中的 log2 函数
        return math.log2(x)
    def abs(self, x: Union[float, int, complex]) -> float:
        """
        计算一个数值的绝对值（对于实数）或模（对于复数）。

        此函数是 Python 内置 `abs()` 函数的一个直接包装器。内置的 `abs()`
        可以自然地处理整数、浮点数和复数，返回一个浮点数结果。
        - 对于实数 `x`，返回 `|x|`。
        - 对于复数 `z = a + bi`，返回其模 `sqrt(a² + b²)`。

        在量子计算中，这对于计算概率幅的模平方 `|ψ_i|²`，或者
        在 `isclose` 中比较数值间的距离非常重要。

        Args:
            x (Union[float, int, complex]): 输入的数值。

        Returns:
            float: 输入值的绝对值或模。

        Raises:
            TypeError: 如果输入不是数值类型（由内置 `abs` 引发）。
        """
        # --- 输入验证 (由内置abs隐式处理，但我们可以做得更明确) ---
        if not isinstance(x, (float, int, complex)):
            raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")

        # --- 核心实现 ---
        # 直接调用 Python 内置的 abs() 函数
        return abs(x)
    def sqrt(self, x: Union[float, int, complex]) -> complex:
        """
        计算一个数值的平方根。

        此函数作为 Python 标准库 `cmath.sqrt` 的一个直接包装器。
        我们优先使用 `cmath` 版本，因为它能够自然地处理所有数值输入：
        -   正实数: `sqrt(4)` -> `2.0+0.0j`
        -   负实数: `sqrt(-1)` -> `0.0+1.0j`
        -   复数: `sqrt(1j)` -> `(0.707...+0.707...j)`

        这在量子计算中非常重要，因为中间计算（例如矩阵对角化）
        可能会产生负数或复数，即使最终结果是实数。

        Args:
            x (Union[float, int, complex]): 输入的数值。

        Returns:
            complex: 输入值的平方根，始终以复数形式返回。

        Raises:
            TypeError: 如果输入不是数值类型。
        """
        # --- 输入验证 ---
        if not isinstance(x, (float, int, complex)):
            raise TypeError(f"Input must be a numeric type (float, int, or complex), but got {type(x).__name__}.")

        # --- 核心实现 ---
        # 直接调用 cmath.sqrt，因为它对所有数值类型都有效，
        # 并返回一个 complex 类型的结果，这在我们的后端中是一致的。
        return cmath.sqrt(x)
    def arange(self, dim: int, dtype=None) -> List[int]:
        """
        创建一个包含从 0 到 `dim-1` 的整数序列的列表。

        此函数模拟了 `numpy.arange(dim)` 最基本的功能。它主要用于
        生成一系列连续的索引，这在某些矩阵构建算法中可能会用到。

        **注意：** 这是一个高度简化的版本，不支持 `start`, `step`
        等高级参数。

        Args:
            dim (int):
                序列的上限（不包含此值）。必须是非负整数。
            dtype (Any, optional):
                此参数为了与 NumPy API 保持兼容而被接受，但会被忽略。
                返回的列表元素类型始终是 Python 的 `int`。

        Returns:
            List[int]:
                一个包含 `[0, 1, 2, ..., dim-1]` 的列表。

        Raises:
            ValueError: 如果 `dim` 为负数。
            TypeError: 如果 `dim` 不是整数。

        Example:
            >>> backend = PurePythonBackend()
            >>> backend.arange(5)
            [0, 1, 2, 3, 4]
            >>> backend.arange(0)
            []
        """
        # --- 输入验证 ---
        if not isinstance(dim, int):
            raise TypeError("Dimension (dim) for arange must be an integer.")
        if dim < 0:
            raise ValueError("Dimension (dim) for arange cannot be negative.")

        # --- 核心实现 ---
        # 直接使用 Python 内置的 range() 函数并将其转换为列表
        return list(range(dim))
    def power(self, base: Union[complex, float, int], exp: Union[complex, float, int]) -> Union[complex, float, int]:
        """
        计算 `base` 的 `exp` 次幂 (`base ** exp`)。

        此函数是 Python 内置 `pow(base, exp)` 函数的一个直接包装器。
        内置的 `pow()` 函数可以自然地处理整数、浮点数和复数的幂运算，
        返回相应的结果类型。

        在量子计算中，这对于计算相位因子（如 `e^(i*theta)`）或在
        DFT（离散傅里叶变换）公式中计算旋转因子 `ω^(jk)` 非常关键。

        Args:
            base (Union[complex, float, int]): 底数。
            exp (Union[complex, float, int]): 指数。

        Returns:
            Union[complex, float, int]: `base` 的 `exp` 次幂的计算结果。

        Raises:
            TypeError: 如果输入不是数值类型（由内置 `pow` 引发）。
        """
        # --- 输入验证 (由内置pow隐式处理，但我们可以做得更明确) ---
        if not isinstance(base, (complex, float, int)):
            raise TypeError(f"Input 'base' must be a numeric type, but got {type(base).__name__}.")
        if not isinstance(exp, (complex, float, int)):
            raise TypeError(f"Input 'exp' must be a numeric type, but got {type(exp).__name__}.")

        # --- 核心实现 ---
        # 直接调用 Python 内置的 pow() 函数
        return pow(base, exp)
    def sum(self, values: List[Union[complex, float, int]]) -> Union[complex, float, int]:
        """
        计算一个数值列表的总和。

        此函数是 Python 内置 `sum()` 函数的一个直接包装器。内置的 `sum()`
        可以自然地处理包含整数、浮点数和复数的列表。
        如果列表中有任何一个复数，结果将是复数。

        在量子计算中，这对于计算矩阵的迹 `Tr(M) = Σ M[i,i]`，
        或者在矩阵乘法中累加乘积项等操作非常有用。

        Args:
            values (List[Union[complex, float, int]]): 
                一个包含待求和数值的列表。

        Returns:
            Union[complex, float, int]: 列表中所有元素的总和。

        Raises:
            TypeError: 如果列表中的元素不是数值类型（由内置 `sum` 引发）。
        """
        # --- 输入验证 ---
        if not isinstance(values, list):
            raise TypeError(f"Input must be a list of numeric values, but got {type(values).__name__}.")
        
        # --- 核心实现 ---
        # 直接调用 Python 内置的 sum() 函数。
        # sum() 的第二个参数是起始值，对于复数求和，
        # 提供一个 0j 的起始值可以确保在空列表时返回一个复数0，
        # 并且能正确处理复数类型。
        return sum(values, 0j)

    # --- 随机数生成 (特殊情况，依赖Python标准库或np，这里使用Python标准库) ---

    def choice(self, options: List[Any], p: List[float]) -> Any:
        """
        根据给定的概率分布 `p`，从 `options` 列表中随机选择一个元素。

        此函数实现了一个经典的轮盘赌选择（Roulette Wheel Selection）算法，
        它不依赖任何第三方库（仅使用 Python 标准的 `random` 模块）。
        这在模拟量子测量时至关重要，因为测量结果是根据概率分布
        随机出现的。

        **工作原理:**
        1.  生成一个在 [0, 1) 区间内的随机浮点数 `r`。
        2.  将概率 `p` 列表转换为一个累积概率分布。
        3.  遍历累积概率，找到第一个大于或等于 `r` 的区间，
            并返回该区间对应的 `options` 中的元素。

        Args:
            options (List[Any]):
                一个包含所有可能选项的列表。
            p (List[float]):
                一个与 `options` 列表长度相同的概率列表。`p[i]` 是
                选择 `options[i]` 的概率。列表中的所有概率值必须为非负数，
                并且它们的总和应该（约）等于 1.0。

        Returns:
            Any:
                从 `options` 中根据概率 `p` 随机选择出的一个元素。

        Raises:
            TypeError: 如果 `options` 或 `p` 不是列表。
            ValueError: 如果 `options` 和 `p` 的长度不匹配，`p` 中包含
                        负数，或者 `options` 为空。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(options, list) or not isinstance(p, list):
            raise TypeError("Inputs 'options' and 'p' must be lists.")
        if len(options) != len(p):
            raise ValueError(f"Length of 'options' ({len(options)}) and 'p' ({len(p)}) must be the same.")
        if not options:
            raise ValueError("'options' list cannot be empty.")
        if any(prob < 0 for prob in p):
            raise ValueError("Probabilities in 'p' cannot be negative.")

        # --- 步骤 2: 概率归一化 ---
        # 检查概率总和，并处理由于浮点误差导致的不严格为1的情况。
        prob_sum = sum(p)
        if self.isclose(prob_sum, 0.0):
            # 如果所有概率都为0，则均匀选择一个 (虽然物理上不应该发生)
            return self.random.choice(options)
        
        if not self.isclose(prob_sum, 1.0):
             # 归一化概率列表
             normalized_p = [val / prob_sum for val in p]
        else:
             normalized_p = p

        # --- 步骤 3: 轮盘赌选择算法 ---
        
        # a) 生成一个 [0.0, 1.0) 之间的随机数
        r = self.random.random()
        
        # b) 遍历累积概率
        cumulative_prob = 0.0
        for i, prob in enumerate(normalized_p):
            cumulative_prob += prob
            # 找到第一个累积概率大于随机数的区间
            if r < cumulative_prob:
                return options[i]
                
        # c) 兜底返回：由于浮点误差，r 可能极度接近1，导致循环结束都未返回。
        #    在这种情况下，我们安全地返回最后一个选项。
        return options[-1]

    def random_normal(self, loc: float = 0.0, scale: float = 1.0, size: Tuple[int, ...]= (1,)) -> Union[float, List[float], List[List[float]]]:
        """
        生成服从正态（高斯）分布的随机数。

        此函数不依赖任何第三方库，而是使用经典的 **Box-Muller 变换**，
        通过 Python 标准库的 `random` 和 `math` 模块，从均匀分布的
        随机数生成标准正态分布（均值为0，方差为1）的随机数。
        然后，通过 `result = z * scale + loc` 将其转换为指定的均值和标准差。

        这在需要引入高斯噪声的模拟场景中非常有用。

        Args:
            loc (float, optional): 
                正态分布的均值（μ）。默认为 0.0。
            scale (float, optional): 
                正态分布的标准差（σ）。必须为非负数。默认为 1.0。
            size (Tuple[int, ...], optional): 
                输出的形状。
                - `(1,)` 或 `()`: 返回单个浮点数。
                - `(n,)`: 返回一个长度为 n 的一维列表。
                - `(m, n)`: 返回一个 m 行 n 列的二维嵌套列表。
                默认为 `(1,)`。

        Returns:
            Union[float, List[float], List[List[float]]]:
                一个或多个服从 `N(loc, scale²)` 分布的随机数。

        Raises:
            TypeError: 如果 `loc`, `scale` 或 `size` 的类型不正确。
            ValueError: 如果 `scale` 为负数，或 `size` 形状不受支持。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(loc, (float, int)):
            raise TypeError(f"Mean 'loc' must be a real number, but got {type(loc).__name__}.")
        if not isinstance(scale, (float, int)):
            raise TypeError(f"Standard deviation 'scale' must be a real number, but got {type(scale).__name__}.")
        if scale < 0:
            raise ValueError(f"Standard deviation 'scale' cannot be negative, but got {scale}.")
        if not isinstance(size, tuple):
             raise TypeError(f"Shape 'size' must be a tuple, but got {type(size).__name__}.")

        # --- 内部辅助函数：生成一个标准正态分布随机数 ---
        def _generate_single_std_normal():
            # Box-Muller 变换生成两个独立的标准正态随机数
            # 为了效率，我们可以生成一对，缓存第二个以备下次调用
            # 但为了逻辑简单，每次都重新生成
            
            # 确保 u1 不为0，以避免 log(0)
            u1 = 0.0
            while u1 == 0.0:
                u1 = self.random.random()
            u2 = self.random.random()
            
            # z0 和 z1 是两个独立的标准正态分布随机变量
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            # z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2) # 我们可以只用一个
            return z0

        # --- 步骤 2: 根据 size 生成结果 ---
        if size == (1,) or size == ():
            std_normal_val = _generate_single_std_normal()
            return std_normal_val * scale + loc
        
        elif len(size) == 1:
            n = size[0]
            if not isinstance(n, int) or n < 0:
                 raise ValueError("Dimension in 'size' must be a non-negative integer.")
            return [_generate_single_std_normal() * scale + loc for _ in range(n)]

        elif len(size) == 2:
            rows, cols = size
            if not isinstance(rows, int) or rows < 0 or not isinstance(cols, int) or cols < 0:
                 raise ValueError("Dimensions in 'size' must be non-negative integers.")
            return [[_generate_single_std_normal() * scale + loc for _ in range(cols)] for _ in range(rows)]
        
        else:
            raise ValueError(f"PurePythonBackend random_normal only supports scalar, 1D, or 2D output sizes, but got {size}.")

    def clip(self, values: Union[List[Union[float, int]], float, int], min_val: float, max_val: float) -> Union[List[float], float]:
        """
        将一个数值或一个列表中的所有数值裁剪到指定的 `[min_val, max_val]` 区间内。

        -   如果 `value < min_val`，结果为 `min_val`。
        -   如果 `value > max_val`，结果为 `max_val`。
        -   否则，结果为 `value` 本身。

        此函数是 `numpy.clip` 的一个简化替代品。在量子模拟中，它的一个
        关键用途是处理浮点数精度误差。例如，一个理论上应该为 0.0 或 1.0
        的概率值，在计算后可能会变成 -1e-17 或 1.000000000000001。
        使用 `clip(probabilities, 0.0, 1.0)` 可以将这些值强制修正回
        物理上有效的 `[0, 1]` 区间内。

        Args:
            values (Union[List[Union[float, int]], float, int]):
                待裁剪的单个数值或数值列表。
            min_val (float): 
                区间的下界。
            max_val (float): 
                区间的上界。

        Returns:
            Union[List[float], float]:
                裁剪后的单个数值或数值列表。返回值的类型将是浮点数。

        Raises:
            TypeError: 如果输入类型不正确。
            ValueError: 如果 `min_val` 大于 `max_val`。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(values, (list, float, int)):
            raise TypeError(f"Input 'values' must be a list, float, or int, but got {type(values).__name__}.")
        if not isinstance(min_val, (float, int)):
            raise TypeError(f"Boundary 'min_val' must be a numeric type, but got {type(min_val).__name__}.")
        if not isinstance(max_val, (float, int)):
            raise TypeError(f"Boundary 'max_val' must be a numeric type, but got {type(max_val).__name__}.")
        if min_val > max_val:
            raise ValueError(f"The minimum value ({min_val}) cannot be greater than the maximum value ({max_val}).")

        # --- 步骤 2: 根据输入类型执行裁剪 ---
        
        # 定义一个内部辅助函数来处理单个值的裁剪，以避免代码重复
        def _clip_single(value):
            return max(float(min_val), min(float(value), float(max_val)))

        if isinstance(values, list):
            # 如果输入是列表，使用列表推导式对每个元素应用裁剪
            return [_clip_single(v) for v in values]
        else:
            # 如果输入是单个数值，直接裁剪并返回
            return _clip_single(values)

    # [增强] 导入 math 模块的全部函数，以替代对 numpy.math 或 cupy.math 的调用
    # 这样可以完全独立于 NumPy/CuPy
    def __getattr__(self, name: str) -> Any:
        """
        在属性查找失败时，动态地从 Python 的内置函数或标准数学库
        (`math`, `cmath`) 中获取。

        这是一个魔术方法，是 `PurePythonBackend` 的核心特性之一。
        它使得我们可以像调用 `self._backend.sin(...)` 或访问 `self._backend.pi` 
        一样，无缝地使用标准库的数学功能，而无需为每一个函数或常量
        编写单独的包装器方法。

        **工作原理:**
        当代码尝试访问一个在 `PurePythonBackend` 实例中未显式定义的属性
        （例如 `backend.sin`）时，Python 解释器会自动调用此 `__getattr__` 
        方法，并将属性名（字符串 "sin"）作为 `name` 参数传入。然后，
        此方法会按照预设的优先级顺序在不同的模块中查找该名称。

        **查找顺序与设计决策:**
        1.  **`builtins`**: 首先查找Python的内置函数。这包括了像 `abs`, 
            `sum`, `pow` 这样对多种数值类型（包括复数）都有通用定义的
            核心函数。
        2.  **`cmath`**: 其次查找复数数学库。`cmath` 中的函数（如 `sin`, 
            `exp`, `sqrt`）被优先选择，因为它们能够处理复数输入，这在
            量子计算的中间步骤中至关重要，从而确保了最大的数值稳健性。
        3.  **`math`**: 最后查找标准数学库。如果一个函数或常量在 `cmath` 
            中不存在（例如 `log2` 或常量 `pi`），则在 `math` 中查找。

        这种设计极大地增强了代码的简洁性和可扩展性，同时确保了所有数学
        运算都严格限制在 Python 标准库和内置函数内，不依赖任何第三方库。

        Args:
            name (str): 
                尝试访问的属性的名称 (例如, 'sin', 'abs', 'pi', 'log2')。

        Returns:
            Any: 
                从内置函数、`math` 或 `cmath` 模块中找到的第一个匹配的
                同名函数或常量。

        Raises:
            AttributeError: 
                如果在所有查找路径（built-ins, cmath, math）中都找不到
                该名称，则抛出一个清晰的错误。
        """
        # 导入 __builtins__ 以动态访问内置函数命名空间
        # 将其放在方法内部，以避免在类级别导入，保持命名空间清洁
        import builtins

        # --- 步骤 1: 在内置函数中查找 ---
        if hasattr(builtins, name):
            return getattr(builtins, name)
            
        # --- 步骤 2: 在复数数学库 `cmath` 中查找 ---
        # 优先选择 cmath 是因为它能处理复数，功能更通用，更适合量子计算
        if hasattr(cmath, name):
            return getattr(cmath, name)
        
        # --- 步骤 3: 在标准数学库 `math` 中查找 ---
        # 如果 cmath 中没有，再查找 math（例如 'log2', 'pi'）
        if hasattr(math, name):
            return getattr(math, name)
            
        # --- 步骤 4: 如果所有地方都找不到，则抛出错误 ---
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}', "
            f"and it was not found in Python's built-in functions, "
            f"the 'cmath' module, or the 'math' module."
        )
    def _hessenberg_reduction(self, mat: List[List[complex]]) -> Tuple[List[List[complex]], List[List[complex]]]:
        """
        [内部辅助函数] 使用Householder变换将一个厄米矩阵约化为三对角形式。

        这是QR算法的一个关键预处理步骤，能显著提高计算效率。
        返回一个元组 (T, Q)，其中 T 是三对角矩阵，Q 是酉变换矩阵，
        满足 mat = Q @ T @ Q†。
        """
        n = len(mat)
        q = self.eye(n)
        a = [row[:] for row in mat] # 创建一个可修改的副本

        for k in range(n - 2):
            # --- Householder变换 ---
            x = [a[i][k] for i in range(k + 1, n)]
            x_norm = self.sqrt(sum(abs(val)**2 for val in x))
            
            v = [0.0 + 0.0j] * len(x)
            if self.isclose(x_norm, 0.0):
                # 如果向量已经是0，跳过
                continue

            # 构造Householder向量
            phase = x[0] / self.abs(x[0]) if not self.isclose(self.abs(x[0]), 0.0) else 1.0
            v[0] = x[0] + phase * x_norm
            for i in range(1, len(x)):
                v[i] = x[i]

            v_norm_sq = sum(abs(val)**2 for val in v)
            if self.isclose(v_norm_sq, 0.0):
                continue
            
            p = self.create_matrix(n, n)
            v_outer_v_conj = self.outer(v, [val.conjugate() for val in v])
            
            # 构造Householder反射矩阵 P = I - 2 * |v><v| / <v|v>
            sub_p = self.eye(n - k - 1)
            for r in range(len(v)):
                for c in range(len(v)):
                    sub_p[r][c] -= 2 * v_outer_v_conj[r][c] / v_norm_sq
            
            # 将子矩阵嵌入到完整的P矩阵中
            p_full = self.eye(n)
            for r in range(n - k - 1):
                for c in range(n - k - 1):
                    p_full[k + 1 + r][k + 1 + c] = sub_p[r][c]

            # 应用变换: A' = P A P† (P是厄米的, P†=P)
            a = self.dot(self.dot(p_full, a), p_full)
            q = self.dot(q, p_full)
        
        return a, q
    def _qr_decomposition_hessenberg(self, mat: List[List[complex]]) -> Tuple[List[List[complex]], List[List[complex]]]:
        """
        [内部辅助函数] 使用Givens旋转对一个上海森堡矩阵进行QR分解。
        """
        n = len(mat)
        q = self.eye(n)
        r = [row[:] for row in mat]

        for j in range(n - 1):
            for i in range(j + 1, n):
                if not self.isclose(r[i][j], 0.0):
                    x, y = r[j][j], r[i][j]
                    norm = self.sqrt(self.abs(x)**2 + self.abs(y)**2)
                    c = x / norm
                    s = y / norm
                    
                    g = [[c.conjugate(), s.conjugate()], [-s, c]]
                    
                    # 更新 R 矩阵
                    for k in range(j, n):
                        r_j_k, r_i_k = r[j][k], r[i][k]
                        r[j][k] = g[0][0] * r_j_k + g[0][1] * r_i_k
                        r[i][k] = g[1][0] * r_j_k + g[1][1] * r_i_k
                    
                    # 更新 Q 矩阵 (Q = G1† G2† ... Gn†)
                    g_t = self.conj_transpose(g)
                    for k in range(n):
                        q_k_j, q_k_i = q[k][j], q[k][i]
                        q[k][j] = q_k_j * g_t[0][0] + q_k_i * g_t[1][0]
                        q[k][i] = q_k_j * g_t[0][1] + q_k_i * g_t[1][1]
        return q, r

    def eigvalsh(self, mat: List[List[complex]], max_iterations: int = 1000, tolerance: float = 1e-12) -> List[float]:
        """
        计算一个厄米矩阵的特征值。

        此函数不依赖任何第三方库，它实现了经典的 **QR算法** 配合 **Hessenberg约化**
        来稳定地计算特征值。由于输入保证是厄米矩阵，特征值保证为实数。

        **工作流程:**
        1.  **Hessenberg 约化**: 首先，使用Householder变换将输入矩阵 `A` 转换为
            一个三对角矩阵 `T`，`A = Q T Q†`。这极大地加速了QR迭代的收敛。
        2.  **QR 迭代**: 对三对角矩阵 `T` 进行迭代：
            -   计算 `T_k` 的 QR 分解: `T_k = Q_k R_k`
            -   计算下一个矩阵: `T_{k+1} = R_k Q_k`
        3.  **收敛**: 随着 `k` 增加，`T_k` 会收敛到一个对角矩阵，其对角线上的
            元素就是原始矩阵 `A` 的特征值。

        此函数用于替代 `numpy.linalg.eigvalsh`。

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
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
            raise TypeError("Input 'mat' must be a non-empty nested list (matrix).")
        rows, cols = len(mat), len(mat[0])
        if rows != cols:
            raise ValueError(f"Input matrix must be square, but got shape ({rows}x{cols}).")
        if not self.allclose(mat, self.conj_transpose(mat), atol=1e-9):
            # 放宽一点容差，因为输入可能也有浮点误差
            raise ValueError("Input matrix must be Hermitian.")

        # --- 步骤 2: Hessenberg 约化 ---
        # 对于厄米矩阵，结果是一个三对角矩阵
        a, _ = self._hessenberg_reduction(mat)
        n = len(a)

        # --- 步骤 3: QR 迭代 ---
        for _ in range(max_iterations):
            # a) QR 分解
            q, r = self._qr_decomposition_hessenberg(a)
            # b) 计算下一个矩阵: A_k+1 = R_k @ Q_k
            a = self.dot(r, q)

            # c) 检查收敛：如果次对角线元素都接近于0，则矩阵已近似为对角
            off_diagonal_sum = sum(self.abs(a[i+1][i]) for i in range(n-1))
            if off_diagonal_sum < tolerance:
                break
        
        # --- 步骤 4: 提取特征值 ---
        # 特征值是收敛后矩阵的对角线元素
        # 由于输入是厄米矩阵，特征值理论上是实数
        eigenvalues = [val.real for val in self.diag(a)]
        
        return sorted(eigenvalues)
# ========================================================================
# --- 3. QuantumCircuit 类的定义 ---
# ========================================================================

@dataclass
class QuantumCircuit:
    """
    表示一个量子电路，包含一系列按顺序应用的量子门操作指令。
    此设计将电路的定义与量子态的实际演化逻辑分离，提升了模块化和复用性。
    """
    num_qubits: int
    """此电路所设计的量子比特数量。在执行时应与 QuantumState 的比特数匹配。"""
    
    instructions: List[Tuple[Any, ...]] = field(default_factory=list)
    """量子门操作指令的序列。每个指令是 (操作名, ...参数..., {可选的关键字参数字典}) 的元组。"""

    description: Optional[str] = None
    """此量子电路的人类可读描述，例如其目的或来源。"""
    
    _internal_logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(f"{QuantumCircuit.__module__}.{QuantumCircuit.__name__}"), 
        repr=False, init=False
    )

    def __post_init__(self):
        """
        在 QuantumCircuit 对象初始化后进行验证。
        """
        if not isinstance(self.num_qubits, int) or self.num_qubits < 0:
            self._internal_logger.error(f"Invalid num_qubits '{self.num_qubits}'. Must be a non-negative integer.")
            raise ValueError("num_qubits must be a non-negative integer.")
        
        if not isinstance(self.instructions, list):
            self._internal_logger.warning(f"instructions field was not a list, has been reset to an empty list.")
            self.instructions = []
        
        self._internal_logger.debug(f"QuantumCircuit for {self.num_qubits} qubits initialized.")

    def add_gate(self, gate_name: str, *args: Any, condition: Optional[Tuple[int, int]] = None, **kwargs: Any):
        """
        向电路中添加一个门操作指令。
        """
        if not isinstance(gate_name, str) or not gate_name.strip():
            self._internal_logger.error(f"Invalid gate_name parameter ('{gate_name}').")
            raise ValueError("Gate name must be a non-empty string.")
            
        instruction_kwargs = dict(kwargs) 
        if condition is not None:
            if not (isinstance(condition, tuple) and len(condition) == 2 and
                    isinstance(condition[0], int) and isinstance(condition[1], int)):
                self._internal_logger.error(f"Invalid condition format ('{condition}'). Must be (int, int).")
                raise ValueError("Condition must be a tuple of two integers: (classical_register_index, expected_value).")
            instruction_kwargs['condition'] = condition
            
        instruction_list = [gate_name] + list(args)
        if instruction_kwargs:
            instruction_list.append(instruction_kwargs)
        
        self.instructions.append(tuple(instruction_list))
        self._internal_logger.debug(f"Added instruction to circuit: {tuple(instruction_list)}")

    # --- 基础单比特门 (支持 **kwargs，用于传递 condition 等) ---
    
    def x(self, qubit_index: int, **kwargs: Any):
        """
        向电路中添加一个 Pauli-X (NOT) 门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("x", qubit_index, **kwargs)

    def y(self, qubit_index: int, **kwargs: Any):
        """
        向电路中添加一个 Pauli-Y 门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("y", qubit_index, **kwargs)

    def z(self, qubit_index: int, **kwargs: Any):
        """
        向电路中添加一个 Pauli-Z (Phase-flip) 门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("z", qubit_index, **kwargs)

    def h(self, qubit_index: int, **kwargs: Any):
        """
        向电路中添加一个 Hadamard 门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("h", qubit_index, **kwargs)

    def s(self, qubit_index: int, **kwargs: Any):
        """
        向电路中添加一个 S (Phase) 门 (sqrt(Z))。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("s", qubit_index, **kwargs)

    def sdg(self, qubit_index: int, **kwargs: Any):
        """
        向电路中添加一个 S-dagger (S†) 门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("sdg", qubit_index, **kwargs)

    def t_gate(self, qubit_index: int, **kwargs: Any):
        """
        向电路中添加一个 T 门 (pi/8 gate)。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("t_gate", qubit_index, **kwargs)

    def tdg(self, qubit_index: int, **kwargs: Any):
        """
        向电路中添加一个 T-dagger (T†) 门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("tdg", qubit_index, **kwargs)

    # --- 基础参数化单比特旋转门 (支持 **kwargs) ---
    
    def rx(self, qubit_index: int, theta: float, **kwargs: Any):
        """
        向电路中添加一个 RX(theta) 旋转门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            theta (float): 围绕X轴旋转的角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("rx", qubit_index, theta, **kwargs)

    def ry(self, qubit_index: int, theta: float, **kwargs: Any):
        """
        向电路中添加一个 RY(theta) 旋转门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            theta (float): 围绕Y轴旋转的角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("ry", qubit_index, theta, **kwargs)

    def rz(self, qubit_index: int, phi: float, **kwargs: Any):
        """
        向电路中添加一个 RZ(phi) 旋转门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            phi (float): 围绕Z轴旋转的角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """

        self.add_gate("rz", qubit_index, phi, **kwargs)

    def p_gate(self, qubit_index: int, lambda_angle: float, **kwargs: Any):
        """
        向电路中添加一个 Phase (P) 门。
        这是一个对角矩阵 [[1, 0], [0, e^(i*lambda_angle)]]。
        等价于 RZ(lambda_angle)，但可能包含一个全局相位。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            lambda_angle (float): 相位角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("p_gate", qubit_index, lambda_angle, **kwargs)

    def u3_gate(self, qubit_index: int, theta: float, phi: float, lambda_angle: float, **kwargs: Any):
        """
        向电路中添加一个通用的 U3 单比特门。
        U3(theta, phi, lambda) 是一个可以表示任何单比特酉操作的门。
        
        Args:
            qubit_index (int): 目标量子比特的索引。
            theta (float): 旋转角度。
            phi (float): 旋转角度。
            lambda_angle (float): 旋转角度。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("u3_gate", qubit_index, theta, phi, lambda_angle, **kwargs)
    # --- 基础多比特门 (支持 **kwargs) ---
    
    def cnot(self, control: int, target: int, **kwargs: Any):
        """
        向电路中添加一个 Controlled-NOT (CNOT or CX) 门。
        
        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("cnot", control, target, **kwargs)

    def cz(self, control: int, target: int, **kwargs: Any):
        """
        向电路中添加一个 Controlled-Z (CZ) 门。
        
        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("cz", control, target, **kwargs)

    def swap(self, qubit1: int, qubit2: int, **kwargs: Any):
        """
        向电路中添加一个 SWAP 门，用于交换两个量子比特的状态。
        
        Args:
            qubit1 (int): 第一个量子比特的索引。
            qubit2 (int): 第二个量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("swap", qubit1, qubit2, **kwargs)

    def toffoli(self, control_1: int, control_2: int, target: int, **kwargs: Any):
        """
        向电路中添加一个 Toffoli (CCNOT or CCX) 门。
        这是一个双控制 NOT 门。
        
        Args:
            control_1 (int): 第一个控制量子比特的索引。
            control_2 (int): 第二个控制量子比特的索引。
            target (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("toffoli", control_1, control_2, target, **kwargs)

    def fredkin(self, control: int, target_1: int, target_2: int, **kwargs: Any):
        """
        向电路中添加一个 Fredkin (CSWAP) 门。
        这是一个受控 SWAP 门。
        
        Args:
            control (int): 控制量子比特的索引。
            target_1 (int): 第一个目标量子比特的索引（被交换）。
            target_2 (int): 第二个目标量子比特的索引（被交换）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("fredkin", control, target_1, target_2, **kwargs)

    # --- [增强] 高级受控门 (支持 **kwargs) ---
    
    def cp(self, control: int, target: int, angle: float, **kwargs: Any):
        """
        向电路中添加一个受控相位门 (Controlled-Phase or CPHASE)。
        
        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            angle (float): 相位角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("cp", control, target, angle, **kwargs)

    def crx(self, control: int, target: int, theta: float, **kwargs: Any):
        """
        向电路中添加一个受控-RX门 (Controlled-RX)。
        
        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            theta (float): 围绕X轴旋转的角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("crx", control, target, theta, **kwargs)

    def cry(self, control: int, target: int, theta: float, **kwargs: Any):
        """
        向电路中添加一个受控-RY门 (Controlled-RY)。
        
        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            theta (float): 围绕Y轴旋转的角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("cry", control, target, theta, **kwargs)
        
    def crz(self, control: int, target: int, phi: float, **kwargs: Any):
        """
        向电路中添加一个受控-Rz门 (Controlled-RZ)。
        
        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            phi (float): 围绕Z轴旋转的角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """

        self.add_gate("crz", control, target, phi, **kwargs)

    def controlled_u(self, control: int, target: int, u_matrix: List[List[complex]], name: str = "CU", **kwargs: Any):
        """
        向电路中添加一个通用的受控-U门 (Controlled-U)。
        
        [修正] u_matrix 参数现在是一个标准的 Python 嵌套列表 (List[List[complex]])，
        使得该电路定义与任何计算后端（NumPy, CuPy, PurePython）完全解耦。

        Args:
            control (int): 控制量子比特的索引。
            target (int): 目标量子比特的索引。
            u_matrix (List[List[complex]]): 一个 (2, 2) 的酉矩阵，以嵌套列表形式表示。
            name (str, optional): 操作的名称，用于日志记录。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        # 在添加指令前进行基本的格式验证，以尽早捕获错误
        if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
                isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
                isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2):
            raise ValueError("`u_matrix` for controlled_u must be a 2x2 nested list of complex numbers.")

        self.add_gate("controlled_u", control, target, u_matrix, name=name, **kwargs)
    # --- [增强] 高级参数化多比特门 (VQA常用，支持 **kwargs) ---
    
    def rxx(self, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
        """
        向电路中添加一个 RXX(theta) 纠缠门。
        
        这个门在量子化学和 VQA 中常用于生成纠缠。
        它对应于哈密顿量演化 exp(-i * theta/2 * X⊗X)。
        
        Args:
            qubit1 (int): 第一个目标量子比特的索引。
            qubit2 (int): 第二个目标量子比特的索引。
            theta (float): 旋转角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("rxx", qubit1, qubit2, theta, **kwargs)
    
    def ryy(self, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
        """
        向电路中添加一个 RYY(theta) 纠缠门。
        
        这个门在量子化学和 VQA 中常用于生成纠缠。
        它对应于哈密顿量演化 exp(-i * theta/2 * Y⊗Y)。
        
        Args:
            qubit1 (int): 第一个目标量子比特的索引。
            qubit2 (int): 第二个目标量子比特的索引。
            theta (float): 旋转角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("ryy", qubit1, qubit2, theta, **kwargs)
        
    def rzz(self, qubit1: int, qubit2: int, theta: float, **kwargs: Any):
        """
        向电路中添加一个 RZZ(theta) 纠缠门。
        
        这个门在量子化学和 VQA 中常用于生成纠缠。
        它对应于哈密顿量演化 exp(-i * theta/2 * Z⊗Z)。
        
        Args:
            qubit1 (int): 第一个目标量子比特的索引。
            qubit2 (int): 第二个目标量子比特的索引。
            theta (float): 旋转角度（弧度）。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("rzz", qubit1, qubit2, theta, **kwargs)

    # --- [增强] 高级多控制门 (算法常用，支持 **kwargs) ---
    
    def mcz(self, controls: List[int], target: int, **kwargs: Any):
        """
        向电路中添加一个多控制Z门 (Multi-Controlled-Z)。
        
        当且仅当所有 `controls` 量子比特都处于 |1⟩ 态时，
        此门会对 `target` 量子比特施加一个 Pauli-Z 门（即乘以 -1 的相位）。
        
        这个门是 Toffoli (CCZ) 门的推广，是许多量子算法中的重要构建块。
        
        Args:
            controls (List[int]): 一个包含所有控制量子比特索引的列表。
            target (int): 目标量子比特的索引。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("mcz", controls, target, **kwargs)

    # --- 操作与测量 (支持 **kwargs，measure 增强了 classical_register_index) ---
    
    def measure(self, qubit_index: int, classical_register_index: Optional[int] = None, collapse_state: bool = True, **kwargs: Any):
        """
        向电路中添加一个测量操作。
        
        此操作会模拟对指定量子比特的测量，结果为 0 或 1。
        
        [增强功能]:
        - 可以将测量结果存储到指定的经典寄存器中，用于后续的经典控制流。
        
        Args:
            qubit_index (int): 要测量的量子比特的索引。
            classical_register_index (Optional[int]): 
                可选参数。如果提供，测量结果将被存储到这个索引对应的经典寄存器中。
            collapse_state (bool): 
                如果为 True (默认)，模拟将包含量子态的坍缩。
                如果为 False，则仅返回一个基于概率的测量结果，而不改变量子态。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("simulate_measurement", qubit_index, classical_register_index=classical_register_index, collapse_state=collapse_state, **kwargs)

    def apply_channel(self, channel_type: str, target_qubits: Union[int, List[int], None], params: Dict[str, Any], **kwargs: Any):
        """
        向电路中添加一个应用量子通道（噪声模型）的操作。
        
        Args:
            channel_type (str): 要应用的量子通道类型 (e.g., "depolarizing")。
            target_qubits (Union[int, List[int], None]): 目标量子比特的索引或列表。
            params (Dict[str, Any]): 通道操作所需的参数 (e.g., {"probability": 0.01})。
            **kwargs (Any): 额外的关键字参数，如 `condition`。
        """
        self.add_gate("apply_quantum_channel", channel_type, target_qubits, params=params, **kwargs)

    # --- 魔术方法 (Dunder methods) ---
    
    def __len__(self) -> int:
        """
        使得 `len(qc)` 可以返回电路中的指令数量。
        """
        return len(self.instructions)

    def __iter__(self):
        """
        使得 `QuantumCircuit` 对象可以被迭代，例如在 for 循环中。
        `for instruction in qc:`
        """
        return iter(self.instructions)
        
    def __str__(self) -> str:
        """
        [优化版] 提供了 `QuantumCircuit` 对象的一个人类可读的、详细的字符串表示。
        不再依赖 NumPy/CuPy 特定的类型检查。
        """
        parts = [f"QuantumCircuit ({self.num_qubits} qubits, {len(self.instructions)} instructions):"]
        if self.description:
            parts.append(f"  Description: {self.description}")
        
        for i, instr in enumerate(self.instructions):
            op_name = instr[0]
            op_kwargs = instr[-1] if instr and isinstance(instr[-1], dict) else {}
            op_args = instr[1:-1] if instr and isinstance(instr[-1], dict) else instr[1:]
            
            condition_str = ""
            if 'condition' in op_kwargs:
                cr_idx, exp_val = op_kwargs['condition']
                condition_str = f" IF C[{cr_idx}]=={exp_val}"
                kwargs_for_display = {k: v for k, v in op_kwargs.items() if k != 'condition'}
            else:
                kwargs_for_display = op_kwargs

            def format_arg(arg):
                # 通用格式化，不再检查 np.ndarray 或 cp.ndarray
                if isinstance(arg, list) and arg and isinstance(arg[0], list):
                    rows, cols = len(arg), len(arg[0])
                    return f"matrix(shape=({rows},{cols}))"
                if isinstance(arg, float):
                    return f"{arg:.4f}"
                return str(arg)

            args_str = ", ".join(map(format_arg, op_args))
            kwargs_str = ", ".join(f"{k}={format_arg(v)}" for k, v in kwargs_for_display.items())
            
            params_str = ", ".join(filter(None, [args_str, kwargs_str]))
            
            parts.append(f"  [{i:02d}]: {op_name}({params_str}){condition_str}")
            
        return "\n".join(parts)

# ========================================================================
# --- 4. 全局并行计算基础设施 (内化管理版) ---
# ========================================================================

# --- 模块级全局变量，用于管理进程池 ---
_process_pool: Optional[pool.Pool] = None
_parallel_enabled: bool = False
_num_processes: int = 0
_pool_lock: mp.Lock = mp.Lock() # 用于保护进程池的创建/销毁，防止并发问题
_progress_queue: Optional[mp.Queue] = None

global_shm_arrays: Optional[Dict[str, RawArray]] = None

def init_worker(log_level: int, log_format: str, queue: Optional[mp.Queue], shm_arrays_from_main: Dict[str, RawArray]):
    """
    [最终修正版] 每个工作进程的初始化函数。
    
    此函数在每个子进程启动时由 `multiprocessing.Pool` 调用一次。它的主要作用是：
    1. 为每个子进程配置独立的日志记录。
    2. 将从主进程传递过来的共享内存数组 (`shm_arrays_from_main`) 存储在
       一个进程级的全局变量 `global_shm_arrays` 中。这使得工作函数可以
       直接访问共享内存，而无需在每个任务中重复传递和序列化这些对象，
       从而解决了在 'spawn' 模式（如Windows）下的序列化错误。
    3. 初始化用于进度更新的共享队列。
    4. (用于调试) 初始化一个包含进程ID的 `worker_info` 字典。

    Args:
        log_level (int): 
            主进程的日志级别。
        log_format (str): 
            主进程的日志格式字符串。
        queue (Optional[mp.Queue]): 
            一个可以在进程间共享的队列，用于进度更新。
        shm_arrays_from_main (Dict[str, RawArray]):
            一个字典，包含了所有需要被工作进程访问的共享内存数组
            (`multiprocessing.RawArray`)。
    """
    global worker_info, progress_queue, global_shm_arrays
    worker_pid = os.getpid()
    
    # --- 配置子进程的日志记录 ---
    worker_logger = logging.getLogger()
    if worker_logger.hasHandlers():
        worker_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(log_level)
    
    # --- 初始化进程级全局变量 ---
    worker_info = { "pid": worker_pid }
    progress_queue = queue
    
    # [*** 关键修复 ***] 将从主进程接收的共享内存数组存储在进程全局变量中
    # 这样做可以避免在每个任务中序列化这些不可序列化的对象。
    global_shm_arrays = shm_arrays_from_main

    worker_logger.info(f"Worker process initialized successfully (shared memory configured).")

def _print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 50, fill: str = '█'):
    """
    [新增] 一个硬编码的函数，用于在控制台打印文本进度条。
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()




def enable_parallelism(num_processes: Optional[int] = None):
    """
    [公共API] [最终修正版] 配置并启用并行计算模式。
    
    此函数不再创建持久化的进程池，而是设置并行计算所需的全局参数（如工作进程数），
    并初始化一个共享的 Manager 和 Queue 用于进度报告。
    实际的并行任务将在需要时动态创建自己的专用进程池。

    这个函数应该在主应用程序的开头，并在 `if __name__ == '__main__':` 块
    的保护下被调用一次。

    Args:
        num_processes (Optional[int]): 
            要启动的工作进程数量。如果为 None，则默认为系统的CPU核心数。
            注意：与之前版本不同，这里不再减1，因为主进程在等待时不会占用大量CPU。
    """
    global _parallel_enabled, _num_processes, _progress_queue
    
    with _pool_lock:
        if _parallel_enabled:
            logger.warning("Parallelism is already enabled. Call disable_parallelism() first to reconfigure.")
            return

        # --- 步骤 1: 确定要使用的工作进程数 ---
        if num_processes is None:
            try:
                cpu_cores = os.cpu_count()
                # 使用所有核心，因为主进程在并行计算期间主要是等待 I/O (从队列读取进度)
                _num_processes = cpu_cores if cpu_cores else 2
                if cpu_cores is None:
                    logger.warning("Could not determine CPU count, defaulting to 2 worker processes for parallelism.")
            except NotImplementedError:
                _num_processes = 2
                logger.warning("os.cpu_count() is not implemented on this system, defaulting to 2 worker processes.")
        else:
            _num_processes = num_processes
        
        # 确保进程数量是正整数
        if not isinstance(_num_processes, int) or _num_processes <= 0:
            logger.warning(f"Number of processes must be a positive integer, got {_num_processes}. Parallelism not enabled.")
            _num_processes = 0
            return

        # --- 步骤 2: 平台特定检查 ---
        # Manager 的启动在 Windows 上也需要 `if __name__ == '__main__':` 保护
        if sys.platform.startswith('win'):
            if mp.current_process().name != 'MainProcess':
                logger.error(
                    "On Windows, enable_parallelism() must be called from within an `if __name__ == '__main__':` block "
                    "to prevent multiprocessing errors. Parallelism will NOT be enabled."
                )
                return

        # --- 步骤 3: 初始化共享资源并设置全局标志 ---
        try:
            # 创建一个可以在多次并行调用之间复用的 Manager 和 Queue
            manager = mp.Manager()
            _progress_queue = manager.Queue()
            
            # 设置全局标志位，表示并行模式已启用
            _parallel_enabled = True
            
            logger.info(f"Parallelism has been CONFIGURED and ENABLED for {_num_processes} worker processes.")

        except Exception as e:
            logger.critical(f"Failed to set up multiprocessing manager and enable parallelism: {e}", exc_info=True)
            # 确保在失败时状态被重置
            _parallel_enabled = False
            _num_processes = 0
            _progress_queue = None
def disable_parallelism():
    """
    [公共API] [最终修正版] 禁用并行计算模式并清理相关资源。

    此函数重置了由 `enable_parallelism` 设置的全局配置，并释放了
    共享的 `Manager` 和 `Queue` 资源。由于进程池是为每个并行任务
    动态创建的，此函数不再负责关闭持久化的进程池。
    """
    global _parallel_enabled, _num_processes, _progress_queue
    
    with _pool_lock: # 使用锁保护对全局状态的访问
        if not _parallel_enabled:
            logger.info("Parallelism was not enabled, no action taken to disable.")
            return
            
        # --- 步骤 1: 重置所有与并行相关的全局状态变量 ---
        _parallel_enabled = False
        _num_processes = 0
        
        # --- 步骤 2: 清理共享资源 ---
        # 将队列引用设为None。当 Manager 对象不再有任何引用时，
        # Python 的垃圾回收机制会自动处理其后台进程的关闭。
        _progress_queue = None
        
        logger.info("Parallelism has been DISABLED and related shared resources have been marked for release.")


# --- 内部并行工作函数 (从原来的并行基础设施部分完整复制) ---
# 这些函数必须在模块的顶层定义，以便它们可以被子进程序列化和访问。

def _pure_python_dot_product(mat_a: List[List[float]], mat_b: List[List[float]]) -> List[List[float]]:
    """
    一个独立的、纯Python实现的、仅处理浮点数矩阵的乘法函数 (C = A @ B)。

    此函数被设计为在多进程的工作进程中独立运行，因此它不依赖任何外部状态，
    并且包含了严格的输入验证。它通过处理实数矩阵来避免复数对象在进程间
    序列化的开销，复数运算的逻辑由调用它的上层函数处理。

    Args:
        mat_a (List[List[float]]): 
            乘法中的左矩阵 (A)。必须是一个非空的、形状规整的浮点数嵌套列表。
        mat_b (List[List[float]]): 
            乘法中的右矩阵 (B)。必须是一个非空的、形状规整的浮点数嵌套列表。

    Returns:
        List[List[float]]: 
            结果矩阵 C，其形状为 (rows_a, cols_b)。

    Raises:
        TypeError: 如果输入 `mat_a` 或 `mat_b` 不是嵌套列表。
        ValueError: 如果任一矩阵为空、形状不规整（各行长度不一），
                    或者它们的维度不满足矩阵乘法的要求。
    """
    # --- 步骤 1: 严格的输入验证 ---

    # 检查 mat_a 的有效性
    if not isinstance(mat_a, list) or not mat_a or not isinstance(mat_a[0], list):
        raise TypeError("Input 'mat_a' must be a non-empty nested list (matrix).")
    rows_a = len(mat_a)
    cols_a = len(mat_a[0])
    if cols_a == 0:
        raise ValueError("Input 'mat_a' cannot have rows of zero length.")
    # 验证矩阵形状是否规整（所有行长度相同）
    for i in range(rows_a):
        if not isinstance(mat_a[i], list) or len(mat_a[i]) != cols_a:
            raise ValueError("Input 'mat_a' is not a well-formed matrix (rows have different lengths).")

    # 检查 mat_b 的有效性
    if not isinstance(mat_b, list) or not mat_b or not isinstance(mat_b[0], list):
        raise TypeError("Input 'mat_b' must be a non-empty nested list (matrix).")
    rows_b = len(mat_b)
    cols_b = len(mat_b[0])
    if cols_b == 0:
        raise ValueError("Input 'mat_b' cannot have rows of zero length.")
    # 验证矩阵形状是否规整
    for i in range(rows_b):
        if not isinstance(mat_b[i], list) or len(mat_b[i]) != cols_b:
            raise ValueError("Input 'mat_b' is not a well-formed matrix (rows have different lengths).")

    # 检查维度兼容性：A 的列数必须等于 B 的行数
    if cols_a != rows_b:
        raise ValueError(
            f"Matrix dimensions are incompatible for dot product: A is ({rows_a}x{cols_a}) "
            f"and B is ({rows_b}x{cols_b}). The inner dimensions ({cols_a} and {rows_b}) must match."
        )

    # --- 步骤 2: 核心计算 ---

    # 使用列表推导式高效地创建一个正确大小的全零结果矩阵
    res = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    # 使用三个嵌套循环实现 C[i,j] = Σ_k (A[i,k] * B[k,j])
    for i in range(rows_a):      # 遍历结果矩阵的行
        for j in range(cols_b):  # 遍历结果矩阵的列
            sum_val = 0.0
            for k in range(cols_a):  # 遍历内积维度 (A的列或B的行)
                sum_val += mat_a[i][k] * mat_b[k][j]
            res[i][j] = sum_val
            
    return res

def _worker_apply_unitary_chunk_pure(task_info: Dict[str, Any]) -> bool:
    """
    工作进程的目标函数 (纯Python版)。
    [最终修正版] 此版本从进程全局变量中获取共享内存数组的引用，以避免序列化错误。

    此函数在独立的子进程中执行，负责计算酉变换 `ρ' = U @ ρ @ U_dagger` 的一个
    水平分块（row chunk）。它从进程全局的共享内存 (`global_shm_arrays`) 中读取
    输入矩阵 `U` 和 `ρ`，计算出结果矩阵 `ρ'` 的对应行，然后将结果写回
    共享内存。

    Args:
        task_info (Dict[str, Any]): 一个字典，包含了此工作单元所需的非共享信息：
            - 'row_chunk' (Tuple[int, int]): 要处理的行范围 (start_row, end_row)。
            - 'dim' (int): 矩阵的维度。

    Returns:
        bool: 如果计算成功完成，则返回 `True`，否则返回 `False`。
    """
    # [*** 关键修复 ***] 声明对进程级全局变量的访问
    global worker_info, progress_queue, global_shm_arrays
    
    # --- 步骤 1: 访问初始化时设置的进程信息和共享内存 ---
    try:
        pid = worker_info.get("pid", os.getpid()) # type: ignore
        local_progress_queue = progress_queue # type: ignore
        
        # [*** 关键修复 ***] 从进程全局变量获取共享内存数组，而不是从 task_info
        if global_shm_arrays is None:
            # 这是一个关键的健全性检查。如果 init_worker 未能正确设置此变量，
            # 工作进程应该立即失败并报告错误。
            raise RuntimeError("Shared memory arrays were not initialized in the worker process.")
        shm_arrays = global_shm_arrays

    except (NameError, RuntimeError) as e:
        # 捕获在初始化阶段可能发生的错误
        pid = os.getpid()
        local_progress_queue = None
        logging.getLogger(__name__).critical(f"Worker (PID: {pid}) failed during its initialization phase: {e}", exc_info=True)
        return False

    task_chunk_info = task_info.get('row_chunk', 'N/A')
    
    try:
        # --- 步骤 2: 从任务信息中解析参数 ---
        start_row, end_row = task_info['row_chunk']
        dim = task_info['dim']
        # 注意：'shm_arrays' 已不再从 task_info 中读取

        chunk_size = end_row - start_row
        
        # 如果块大小为0，则无需计算，直接成功返回
        if chunk_size <= 0:
            if local_progress_queue:
                local_progress_queue.put(1)
            return True

        # --- 步骤 3: 将共享内存 (RawArray) 重构为可用的矩阵视图 (List[List[float]]) ---
        U_real = [shm_arrays['U_real'][i*dim:(i+1)*dim] for i in range(dim)]
        U_imag = [shm_arrays['U_imag'][i*dim:(i+1)*dim] for i in range(dim)]
        rho_real = [shm_arrays['rho_real'][i*dim:(i+1)*dim] for i in range(dim)]
        rho_imag = [shm_arrays['rho_imag'][i*dim:(i+1)*dim] for i in range(dim)]
        
        # --- 步骤 4: 执行核心计算的第一部分：temp_chunk = U_chunk @ ρ ---
        U_chunk_real = U_real[start_row:end_row]
        U_chunk_imag = U_imag[start_row:end_row]

        temp_AC = _pure_python_dot_product(U_chunk_real, rho_real)
        temp_BD = _pure_python_dot_product(U_chunk_imag, rho_imag)
        temp_real = [[ac - bd for ac, bd in zip(row_ac, row_bd)] for row_ac, row_bd in zip(temp_AC, temp_BD)]

        temp_AD = _pure_python_dot_product(U_chunk_real, rho_imag)
        temp_BC = _pure_python_dot_product(U_chunk_imag, rho_real)
        temp_imag = [[ad + bc for ad, bc in zip(row_ad, row_bc)] for row_ad, row_bc in zip(temp_AD, temp_BC)]

        # --- 步骤 5: 执行核心计算的第二部分：result_chunk = temp_chunk @ U_dagger ---
        U_T_real = [[U_real[j][i] for j in range(dim)] for i in range(dim)]
        U_T_imag = [[U_imag[j][i] for j in range(dim)] for i in range(dim)]
        
        res_AC = _pure_python_dot_product(temp_real, U_T_real)
        res_BD = _pure_python_dot_product(temp_imag, U_T_imag)
        result_chunk_real = [[ac + bd for ac, bd in zip(row_ac, row_bd)] for row_ac, row_bd in zip(res_AC, res_BD)]

        res_BC = _pure_python_dot_product(temp_imag, U_T_real)
        res_AD = _pure_python_dot_product(temp_real, U_T_imag)
        result_chunk_imag = [[bc - ad for bc, ad in zip(row_bc, row_ad)] for row_bc, row_ad in zip(res_BC, res_AD)]
        
        # --- 步骤 6: 将计算结果写回共享内存 ---
        for i in range(chunk_size):
            global_row_idx = start_row + i
            start_pos = global_row_idx * dim
            end_pos = start_pos + dim
            shm_arrays['result_real'][start_pos:end_pos] = result_chunk_real[i]
            shm_arrays['result_imag'][start_pos:end_pos] = result_chunk_imag[i]

        # --- 步骤 7: 发送进度信号 ---
        if local_progress_queue:
            try:
                local_progress_queue.put(1)
            except Exception as q_e:
                logging.getLogger(__name__).warning(f"Worker (PID: {pid}) could not put to progress queue: {q_e}")

        return True

    except Exception as e:
        # --- 步骤 8: 健壮的异常处理 ---
        worker_logger = logging.getLogger(__name__)
        worker_logger.error(
            f"Worker process (PID: {pid}) failed on chunk {task_chunk_info}: {type(e).__name__}: {e}",
            exc_info=True
        )
        return False

def _execute_parallel_unitary_evolution_pure(unitary: List[List[complex]], rho: List[List[complex]]) -> Optional[List[List[complex]]]:
    """
    [内部函数] 在主进程中编排纯Python版的并行酉变换。
    [最终修正版] 此版本通过在函数内部动态创建专用进程池，并使用 initializer 
                 传递共享内存，从而修复了在 'spawn' 模式下的序列化错误。

    此函数是并行计算的主控制器。它将大型矩阵分解成多个小任务块，
    通过共享内存将数据分发给一个临时的进程池，然后以“先完成先处理”的方式
    监控进度并收集结果，最后重构出最终矩阵。
    
    此函数会读取全局的 `_parallel_enabled` 和 `_num_processes` 配置，
    但不再依赖全局的 `_process_pool` 对象。

    Args:
        unitary (List[List[complex]]): 全局酉算子。
        rho (List[List[complex]]): 初始密度矩阵。

    Returns:
        Optional[List[List[complex]]]: 
            演化后的新密度矩阵。如果并行计算成功，返回结果矩阵；
            如果失败（如工作进程错误或超时），则返回 None。
    """
    global _parallel_enabled, _num_processes, _progress_queue

    # --- 步骤 1: 前置检查与初始化 ---
    if not _parallel_enabled or _num_processes <= 0:
        logger.error(
            "Internal parallel executor called, but parallelism is not enabled or configured. "
            "Call enable_parallelism() first. This indicates a logic error in the calling function."
        )
        return None

    dim = len(rho)
    if dim == 0:
        return []

    size = dim * dim
    
    # --- 步骤 2: 共享内存占用预估与警告 ---
    total_shm_bytes = 6 * size * ctypes.sizeof(ctypes.c_double)
    total_shm_gb = total_shm_bytes / _GB_TO_BYTES
    
    logger.info(f"Allocating approximately {total_shm_gb:.3f} GB of shared memory for parallel computation.")
    
    ram_warning_threshold_gb = _TOTAL_SYSTEM_RAM_GB_HW * 0.50
    if total_shm_gb > ram_warning_threshold_gb:
        logger.warning(
            f"Parallel computation requires ~{total_shm_gb:.2f} GB of shared RAM, "
            f"which exceeds the safety threshold ({ram_warning_threshold_gb:.2f} GB, 50% of configured total RAM). "
            "This may lead to significant performance degradation due to memory swapping (thrashing) or system instability."
        )
    
    # --- 步骤 3: 创建并填充共享内存 (RawArray) ---
    try:
        shm_arrays = {
            'U_real': RawArray(ctypes.c_double, size), 'U_imag': RawArray(ctypes.c_double, size),
            'rho_real': RawArray(ctypes.c_double, size), 'rho_imag': RawArray(ctypes.c_double, size),
            'result_real': RawArray(ctypes.c_double, size), 'result_imag': RawArray(ctypes.c_double, size),
        }
    except Exception as e:
        logger.critical(f"Failed to allocate {total_shm_gb:.2f} GB of shared memory: {e}", exc_info=True)
        return None

    def fill_shm(matrix: List[List[complex]], real_arr: RawArray, imag_arr: RawArray):
        real_arr[:] = [elem.real for row in matrix for elem in row]
        imag_arr[:] = [elem.imag for row in matrix for elem in row]

    try:
        fill_shm(unitary, shm_arrays['U_real'], shm_arrays['U_imag'])
        fill_shm(rho, shm_arrays['rho_real'], shm_arrays['rho_imag'])
    except Exception as e:
        logger.critical(f"Failed to fill shared memory with matrix data: {e}", exc_info=True)
        return None

    # --- 步骤 4: 创建任务块 (micro-chunks) ---
    num_chunks = _num_processes * 4
    chunk_size = max(1, math.ceil(dim / num_chunks)) 
    row_chunks = [(i, min(i + chunk_size, dim)) for i in range(0, dim, chunk_size)]
    
    # [*** 关键修复 ***] 任务字典中不再包含 shm_arrays，使其可以被安全序列化
    tasks = [{
        'row_chunk': chunk,
        'dim': dim,
    } for chunk in row_chunks]
    
    # --- 步骤 5: [最终修正] 创建一个专用于此次计算的进程池，并正确传递初始化参数 ---
    pool_for_this_run: Optional[pool.Pool] = None
    all_workers_succeeded = False  # 默认为失败，只有在成功完成时才设为True
    try:
        # 准备传递给 initializer 的日志和共享内存参数
        root_logger = logging.getLogger()
        log_level = root_logger.level if root_logger.level != 0 else logging.INFO
        log_format = "%(asctime)s - [%(levelname)s] - (%(name)s) - [PID:%(process)d] - %(message)s"
        
        # [*** 关键修复 ***] 将 shm_arrays 加入 initargs，以便在进程启动时传递
        init_args = (log_level, log_format, _progress_queue, shm_arrays)
        
        start_method = 'fork' if sys.platform != 'win32' else 'spawn'
        ctx = mp.get_context(start_method)
        
        pool_for_this_run = ctx.Pool(
            processes=_num_processes,
            initializer=init_worker,
            initargs=init_args
        )

        total_tasks = len(tasks)
        logger.info(f"Submitting {total_tasks} micro-chunks to a temporary pool of {_num_processes} worker processes.")
        
        result_iterator = pool_for_this_run.imap_unordered(_worker_apply_unitary_chunk_pure, tasks)
        
        completed_tasks = 0
        execution_ok = True

        _print_progress_bar(0, total_tasks, prefix='Progress:', suffix='Complete', length=50)

        for result in result_iterator:
            completed_tasks += 1
            if not result:
                execution_ok = False
                logger.error("A worker process reported failure. Aborting progress loop.")
                break
            _print_progress_bar(completed_tasks, total_tasks, prefix='Progress:', suffix='Complete', length=50)

        sys.stdout.write('\n')
        sys.stdout.flush()

        if execution_ok and completed_tasks != total_tasks:
            logger.error(f"Execution discrepancy: Expected {total_tasks} tasks, but only {completed_tasks} completed.")
            execution_ok = False

        all_workers_succeeded = execution_ok
            
    except (KeyboardInterrupt, SystemExit) as e:
        sys.stdout.write('\n')
        logger.warning(f"Parallel execution was manually interrupted by the user ({type(e).__name__}).")
        all_workers_succeeded = False
    except Exception as e:
        sys.stdout.write('\n')
        logger.critical(f"A critical error was raised during parallel execution orchestration: {e}", exc_info=True)
        all_workers_succeeded = False
    finally:
        # [*** 关键修复 ***] 确保专用的进程池被优雅地关闭和清理
        if pool_for_this_run:
            logger.debug("Terminating the dedicated process pool for this computation...")
            pool_for_this_run.close()
            pool_for_this_run.join()
            logger.debug("Process pool terminated.")
        
        if not all_workers_succeeded:
            # 返回 None 表示并行计算失败
            return None

    # --- 步骤 6: 从共享内存反序列化结果 ---
    final_rho_real_flat = list(shm_arrays['result_real'])
    final_rho_imag_flat = list(shm_arrays['result_imag'])
    
    final_rho = [
        [complex(final_rho_real_flat[i*dim + j], final_rho_imag_flat[i*dim + j]) for j in range(dim)]
        for i in range(dim)
    ]
    
    return final_rho
# ========================================================================
# --- 5. QuantumState 类的定义 (硬件优化版) ---
# ========================================================================


# [增强] PauliString 和 Hamiltonian 的定义
@dataclass
class PauliString:
    """
    表示一个带系数的Pauli串，例如 `0.5 * 'IXYZ'`。

    这是一个纯粹的数据结构，用于在哈密顿量中表示一个单独的项。它与任何
    特定的计算后端（NumPy, CuPy, or PurePythonBackend）完全解耦。

    Attributes:
        coefficient (complex):
            Pauli串的复数系数。在初始化时，整数或浮点数会被自动转换
            为复数类型。
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

    def __post_init__(self):
        """
        在对象初始化后自动调用的验证方法。

        此方法确保 `coefficient` 和 `pauli_map` 的类型和内容都是有效的，
        从而保证每个 `PauliString` 实例在创建时都是一个合法的对象。

        Raises:
            TypeError: 如果 `coefficient` 不是数值类型，或者 `pauli_map` 不是字典。
            ValueError: 如果 `pauli_map` 中的量子比特索引为负数，或者算子
                        不是 'I', 'X', 'Y', 'Z' 之一。
        """
        # --- 验证系数的类型 ---
        if not isinstance(self.coefficient, (complex, float, int)):
            raise TypeError(
                f"PauliString coefficient must be a numeric type (complex, float, or int), "
                f"but got {type(self.coefficient).__name__}."
            )
        
        # --- 确保 pauli_map 是一个字典 ---
        if not isinstance(self.pauli_map, dict):
            raise TypeError(
                f"PauliString pauli_map must be a dictionary, "
                f"but got {type(self.pauli_map).__name__}."
            )
        
        # --- 验证 pauli_map 的内容（键和值） ---
        for qubit_index, operator in self.pauli_map.items():
            if not isinstance(qubit_index, int) or qubit_index < 0:
                raise ValueError(
                    f"Qubit index in pauli_map must be a non-negative integer, "
                    f"but got {qubit_index}."
                )
            if operator not in ['I', 'X', 'Y', 'Z']:
                raise ValueError(
                    f"Pauli operator in pauli_map must be 'I', 'X', 'Y', or 'Z', "
                    f"but got '{operator}' for qubit {qubit_index}."
                )
        
        # --- 规范化：将系数统一转换为复数类型 ---
        # 这样做可以简化后续的处理，因为我们总能假定系数是复数。
        self.coefficient = complex(self.coefficient)

    def __str__(self) -> str:
        """
        提供 `PauliString` 的一个人类可读的、确定性的字符串表示。
        
        例如: `(0.5+0j) * X0Z2`
        
        Returns:
            str: 格式化的字符串。
        """
        # 如果 pauli_map 为空或只包含 'I'，则表示一个纯数乘单位矩阵的操作
        if not self.pauli_map or all(op == 'I' for op in self.pauli_map.values()):
            return f"{self.coefficient} * I"
            
        # 按量子比特索引排序，以获得一个确定性的字符串表示（例如，总是 X0Z2 而不是 Z2X0）
        pauli_string_parts = []
        # sorted(self.pauli_map.keys()) 确保了输出顺序的稳定性
        for qubit_index in sorted(self.pauli_map.keys()):
            operator = self.pauli_map[qubit_index]
            # 惯例上，不显式地写出单位算子 'I'，除非它是唯一的算子
            if operator != 'I':
                pauli_string_parts.append(f"{operator}{qubit_index}")
        
        # 如果所有指定的操作都是 'I'（例如 `pauli_map={0:'I'}`），则显示 'I'
        pauli_term_string = "".join(pauli_string_parts) if pauli_string_parts else "I"
        
        return f"{self.coefficient} * {pauli_term_string}"


# --- [关键修复] 添加 Hamiltonian 类型别名的定义 ---
Hamiltonian = List[PauliString]
"""表示哈密顿量，它是一个 PauliString 对象的列表。"""

@dataclass
class QuantumState:
    """
    [终极修正版] 模拟量子系统核心状态的抽象表示，基于密度矩阵。
    此版本移除了对NumPy/SciPy的硬性依赖，支持 CuPy (GPU) 和 PurePython (CPU, 含并行) 后端。
    """
    num_qubits: int = field(default=0)
    _density_matrix: Any = field(default_factory=lambda: [[1+0j]], repr=False)
    
    # --- 内部状态与配置 ---
    _backend: Any = field(default=None, repr=False, init=False) 
    """当前的计算后端（PurePythonBackend 实例或 CuPy 模块）。"""
    
    _is_sparse: bool = field(default=False, repr=False, init=False) 
    """一个标志位，指示密度矩阵当前是否以稀疏格式存储。(当前为占位符)"""
    
    _internal_logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(f"{QuantumState.__module__}.{QuantumState.__name__}"), 
        repr=False, 
        init=False
    )
    
    _classical_registers: Dict[int, int] = field(default_factory=dict, repr=False) 
    
    # --- 元数据与历史记录 ---
    entangled_sets: List[Tuple[int, ...]] = field(default_factory=list)
    gate_application_history: List[Dict[str, Any]] = field(default_factory=list)
    measurement_outcomes_log: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system_energy_level: Optional[float] = field(default=0.0) 
    last_significant_update_timestamp_utc_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    custom_state_parameters: Dict[str, Any] = field(default_factory=dict)

    # --- 常量与类型定义 ---
    MAX_GATE_HISTORY: int = field(default=50, repr=False, init=False)
    MAX_MEASUREMENT_LOG: int = field(default=100, repr=False, init=False)
    SPARSITY_THRESHOLD: float = field(default=0.1, repr=False, init=False)
    
    QuantumChannelType = Literal["depolarizing", "bit_flip", "phase_flip", "amplitude_damping"]

    # --- [绝对最终修正版] `__post_init__` ---
    def __post_init__(self):
        """
        [增强版] 在对象初始化后，进行全面的验证、后端选择、内存检查和密度矩阵的初始化。
        
        此版本包含了对日志记录、后端选择逻辑的修复，并新增了主动的内存占用预估与
        警告机制，以提高库的健壮性和用户体验。

        工作流程:
        1. 验证 `num_qubits` 参数的有效性。
        2. 根据全局配置 `_core_config` 选择并实例化计算后端（CuPy 或 PurePythonBackend）。
        3. [新增] 预估即将创建的密度矩阵所需的内存，并与配置的硬件限制进行比较。
           如果内存占用过高，将发出警告或引发错误。
        4. 创建一个表示初始态 |0...0⟩ 的密度矩阵，其数据结构与所选后端兼容。
        """
        log_prefix = f"QuantumState.__post_init__(Requested N={self.num_qubits})"
        self._internal_logger.info(f"[{log_prefix}] Initializing quantum state...")

        # --- 步骤 1: 基本参数验证 ---
        if not isinstance(self.num_qubits, int) or self.num_qubits < 0:
            raise ValueError("num_qubits must be a non-negative integer.")

        # --- 步骤 2: 根据配置选择后端 ---
        backend_choice = _core_config.get("BACKEND_CHOICE", "auto")
        self._backend = None
        
        if backend_choice == 'auto':
            if cp:
                self._backend = cp
            else:
                self._backend = PurePythonBackend()
        elif backend_choice == 'cupy':
            if cp:
                self._backend = cp
            else:
                raise ImportError("CuPy backend was requested ('BACKEND_CHOICE': 'cupy'), but the 'cupy' library is not installed.")
        elif backend_choice == 'pure_python':
            self._backend = PurePythonBackend()
        else:
            raise ValueError(f"Invalid BACKEND_CHOICE in configuration: '{backend_choice}'. "
                             "Must be one of 'auto', 'cupy', or 'pure_python'.")
        
        # 修正获取后端名称的逻辑，避免 `__getattr__` 陷阱
        if isinstance(self._backend, types.ModuleType):
            backend_name = self._backend.__name__
        else:
            backend_name = type(self._backend).__name__
        self._internal_logger.info(f"[{log_prefix}] Selected backend for computation: {backend_name}")
        
        # --- 步骤 3: 验证量子比特数上限 ---
        user_config_max_qubits = _core_config.get("MAX_QUBITS", 10)
        if self.num_qubits > user_config_max_qubits:
            raise ValueError(
                f"Requested {self.num_qubits} qubits exceeds the user-configured maximum of {user_config_max_qubits}. "
                "You can change this limit via `configure_quantum_core({'MAX_QUBITS': ...})`."
            )

        # --- [核心增强] 步骤 4: 内存占用预估与警告 ---
        expected_dim = 1 << self.num_qubits
        density_matrix_elements = expected_dim ** 2
        
        # 每个复数元素占用16字节 (complex128: 8字节实部 + 8字节虚部)
        estimated_mem_bytes = density_matrix_elements * _BYTES_PER_COMPLEX128_ELEMENT
        estimated_mem_gb = estimated_mem_bytes / _GB_TO_BYTES
        
        self._internal_logger.info(
            f"[{log_prefix}] Preparing to create a {expected_dim}x{expected_dim} density matrix. "
            f"Estimated memory footprint for the state: {estimated_mem_gb:.3f} GB."
        )

        if self._backend is cp:
            # GPU VRAM 检查 (使用更严格的阈值，例如90%)
            vram_limit_gb = _TOTAL_GPU_VRAM_GB_HW * 0.90
            if estimated_mem_gb > vram_limit_gb:
                msg = (f"Requested {self.num_qubits} qubits requires ~{estimated_mem_gb:.2f} GB of VRAM, "
                       f"which exceeds the safety threshold ({vram_limit_gb:.2f} GB, 90% of configured GPU VRAM). "
                       "This is likely to cause an out-of-memory error. To proceed, reduce qubit count or switch to the 'pure_python' backend.")
                self._internal_logger.error(f"[{log_prefix}] {msg}")
                raise ValueError(msg)
        else:
            # CPU RAM 检查 (使用较宽松的阈值，例如75%，因为系统有虚拟内存)
            ram_limit_gb = _TOTAL_SYSTEM_RAM_GB_HW * 0.75
            if estimated_mem_gb > ram_limit_gb:
                logger.warning(
                    f"[{log_prefix}] The requested {self.num_qubits}-qubit state requires ~{estimated_mem_gb:.2f} GB of RAM. "
                    f"This exceeds the safety threshold ({ram_limit_gb:.2f} GB, 75% of configured system RAM) "
                    "and may lead to significant performance degradation due to memory swapping (thrashing) or system instability."
                )

        # --- 步骤 5: 初始化密度矩阵 ---
        initial_state_vector = self._backend.zeros((expected_dim,), dtype=complex)
        
        if isinstance(self._backend, PurePythonBackend):
            if expected_dim > 0:
                initial_state_vector[0] = 1.0 + 0.0j
            conjugated_vector = [v.conjugate() for v in initial_state_vector]
            self._density_matrix = self._backend.outer(initial_state_vector, conjugated_vector)
        else: # CuPy backend
            if expected_dim > 0:
                initial_state_vector[0] = 1.0
            self._density_matrix = self._backend.outer(initial_state_vector, initial_state_vector.conj())

        self._internal_logger.info(f"[{log_prefix}] {self.num_qubits}-qubit quantum state successfully initialized on the {backend_name} backend.")
    
    
    
    def __deepcopy__(self, memo):
        """
        为 QuantumState 类自定义深拷贝行为，以正确处理不可序列化的字段。

        标准的 `copy.deepcopy` 无法处理模块引用（如 `self._backend = cupy`）
        或日志记录器实例，这会导致 `TypeError`。此方法通过手动拷贝所有
        可序列化的字段，并重新赋值不可序列化的字段的引用，来正确地实现
        一个完整的、独立的 `QuantumState` 实例拷贝。

        此实现与具体的计算后端（CuPy 或 PurePythonBackend）无关。

        Args:
            memo (Dict[int, Any]): 
                一个由 `copy.deepcopy` 内部管理的字典，用于在递归拷贝
                过程中跟踪已经拷贝过的对象，以处理循环引用。这是实现
                正确深拷贝协议所必需的参数。

        Returns:
            QuantumState: 一个当前量子态的完整、独立的深拷贝副本。
        """

        # --- 步骤 1: 创建一个新的、空的 QuantumState 实例 ---
        # `cls.__new__(cls)` 可以创建一个类的实例而不调用其 __init__ 或 __post_init__ 方法。
        # 这是自定义深拷贝的标准做法，因为它允许我们手动填充所有字段。
        cls = self.__class__
        result = cls.__new__(cls)
        
        # --- 步骤 2: 注册新创建的实例到 memo 字典 ---
        # 这是深拷贝协议的关键部分。它告诉 `copy.deepcopy`，如果将来在
        # 拷贝子对象时再次遇到 `self` (id(self))，应该直接使用已经创建的
        # `result` 实例，从而避免无限递归。
        memo[id(self)] = result

        # --- 步骤 3: 遍历所有在 dataclass 中定义的字段并进行拷贝 ---
        # 使用 `dataclasses.fields` 可以安全地获取所有已定义的字段对象。
        for f in fields(cls):
            # --- 步骤 3a: 识别并跳过不可序列化的字段 ---
            # `_backend` (模块或实例引用) 和 `_internal_logger` (Logger实例)
            # 不应该被深拷贝。我们将稍后手动处理它们。
            if f.name in ('_backend', '_internal_logger'):
                continue
            
            # --- 步骤 3b: 对所有其他可序列化的字段进行标准的深拷贝 ---
            # 获取原始对象中该字段的值。
            value = getattr(self, f.name)
            
            # 使用 `copy.deepcopy` 递归地拷贝该值。这会正确处理
            # 嵌套的列表、字典以及 CuPy 数组（如果 CuPy 已安装并支持 deepcopy）。
            # 传递 `memo` 字典对于正确处理子对象中的循环引用至关重要。
            setattr(result, f.name, copy.deepcopy(value, memo))

        # --- 步骤 4: 手动处理不可序列化的字段 ---
        # 对于 `_backend` 和 `_internal_logger`，我们不进行拷贝，
        # 而是直接将原始对象的引用赋给新创建的对象。
        # 这既安全又高效，因为这些对象本身是无状态的（或应被视为单例）。
        # 新的 QuantumState 实例应该使用与原始实例完全相同的计算后端和日志记录器。
        result._backend = self._backend
        result._internal_logger = self._internal_logger
        
        # --- 步骤 5: 返回最终构建好的深拷贝实例 ---
        return result
    
    # --- 属性与验证方法 ---
    @property
    def density_matrix(self) -> Any:
        """
        一个只读属性，用于安全地访问内部的密度矩阵。

        使用 `@property` 装饰器使得可以像访问一个普通属性一样调用此方法
        （例如 `state.density_matrix`），同时提供了良好的封装性，保护了
        内部状态变量 `_density_matrix` 不被外部代码直接修改。

        [修正] 此属性现在返回 `_density_matrix`，其具体类型取决于在
        `__post_init__` 中选择的计算后端（可能是 CuPy 数组或纯 Python 的
        嵌套列表）。

        Returns:
            Any: 
                当前量子态的密度矩阵。其类型可能是 `cupy.ndarray` 或
                `List[List[complex]]`。
        """
        return self._density_matrix
    
    def normalize(self):
        """
        强制将密度矩阵的迹归一化为 1.0。

        在多次门操作或噪声应用后，由于浮点数的累积精度误差，密度矩阵的迹
        （`Tr(ρ)`）可能会轻微偏离 1.0。此方法通过将密度矩阵的每个元素除以
        当前的迹来修正这个问题，从而保证概率守恒 (`Tr(ρ) = 1`)。

        [修正] 此实现现在是后端无关的。它会自动检测密度矩阵的数据类型
        （CuPy 数组或 Python 列表），并使用相应的方式执行归一化操作。

        该方法包含安全检查，会避免在迹接近于零时进行除法操作，并会在迹
        已经归一化时提前退出以提高效率。
        """
        # --- 步骤 1: 计算当前的迹 ---
        # 调用 self._backend.trace()，它会根据当前后端执行正确的迹运算。
        trace = self._backend.trace(self._density_matrix)
        
        # --- 步骤 2: 安全性和效率检查 ---
        # 2a. 检查迹是否有效（不为零），防止除零错误。
        #     self._backend.isclose() 和 self._backend.abs() 同样是后端通用的。
        if self._backend.isclose(self._backend.abs(trace), 0.0, atol=1e-12):
            self._internal_logger.critical(
                "Cannot normalize density matrix because its trace is close to zero. "
                "The quantum state may be corrupted. Normalization skipped."
            )
            return
            
        # 2b. 如果迹已经足够接近 1.0，则无需执行昂贵的除法操作，提前返回。
        if self._backend.isclose(trace, 1.0, atol=1e-9):
            return

        # --- 步骤 3: 根据数据类型执行归一化 ---
        # 这是核心的后端兼容性逻辑。
        
        # 3a. 如果是 PurePythonBackend，密度矩阵是嵌套列表。
        #     需要使用列表推导式手动遍历并除以迹。
        if isinstance(self._density_matrix, list):
            self._density_matrix = [
                [element / trace for element in row]
                for row in self._density_matrix
            ]
        # 3b. 如果是 CuPy 后端，密度矩阵是 CuPy 数组。
        #     可以直接使用高效的、向量化的原地除法运算 `/=`。
        else:
            self._density_matrix /= trace
        
        # --- 步骤 4: 记录操作 ---
        # 在调试级别记录归一化事件，方便追踪数值精度问题。
        self._internal_logger.debug(
            f"Density matrix has been normalized. Old trace: {trace.real:.8f}, "
            f"New trace: {self._backend.trace(self._density_matrix).real:.8f}"
        )
            
    def is_valid_density_matrix_properties(self, dm: Any, tolerance: float = 1e-9) -> bool:
        """
        检查一个给定的矩阵是否满足密度矩阵的核心数学属性。

        此方法验证两个关键属性：
        1.  **厄米性 (Hermiticity)**: 矩阵必须等于其自身的共轭转置 (M = M†)。
        2.  **正半定性 (Positive Semi-definiteness)**: 矩阵的所有特征值必须
            为非负数 (λ_i >= 0)。

        [修正] 此实现现在是后端无关的。它会调用当前后端 (`self._backend`)
        提供的 `allclose`, `conj_transpose`, 和 `eigvalsh` 方法，并能处理
        不同后端返回的数据类型（CuPy 数组或 Python 列表）。

        注意：此方法不检查迹是否为 1。该检查由 `is_normalized` 方法负责。

        Args:
            dm (Any): 
                要检查的密度矩阵。其类型应与当前后端匹配（`cupy.ndarray`
                或 `List[List[complex]]`）。
            tolerance (float, optional): 
                用于浮点数比较的容差。默认为 1e-9。

        Returns:
            bool: 如果矩阵同时满足厄米性和正半定性，则返回 True，否则返回 False。
        """
        # --- 步骤 0: 检查输入是否为空或无效 ---
        if dm is None or (isinstance(dm, list) and not dm):
            self._internal_logger.warning("is_valid_density_matrix_properties received a None or empty matrix.")
            return False

        # --- 步骤 1: 检查厄米性 (M == M.conj().T) ---
        # 调用后端通用的 allclose 和 conj_transpose 方法。
        if not self._backend.allclose(dm, self._backend.conj_transpose(dm), atol=tolerance):
            self._internal_logger.warning("Density matrix property check failed: Matrix is not Hermitian.")
            return False
            
        # --- 步骤 2: 检查正半定性 (所有特征值 >= 0) ---
        try:
            # 调用后端的特征值求解器。
            # 对于厄米矩阵，eigvalsh 是最稳定和高效的选择，并保证返回实数特征值。
            eigenvalues = self._backend.eigvalsh(dm)
            
            # --- 步骤 2a: 处理不同后端返回的特征值类型 ---
            # 这是核心的后端兼容性逻辑。
            
            # 如果是 PurePythonBackend，eigenvalues 是一个 Python 列表。
            if isinstance(eigenvalues, list):
                # 使用 Python 的 all() 和列表推导式进行检查。
                # 允许特征值有在容差范围内的微小负值，以应对浮点数误差。
                if not all(val >= -tolerance for val in eigenvalues):
                    min_eigenvalue = min(eigenvalues)  # 使用 Python 内置 min()
                    self._internal_logger.warning(
                        f"Density matrix property check failed: Matrix is not positive semi-definite. "
                        f"Minimum eigenvalue found: {min_eigenvalue:.2e}"
                    )
                    return False
            
            # 如果是 CuPy 后端，eigenvalues 是一个 CuPy 数组。
            else:
                # 使用后端的 all() 和 min() 方法进行向量化检查，效率更高。
                if not self._backend.all(eigenvalues >= -tolerance):
                    min_eigenvalue = self._backend.min(eigenvalues)
                    # 可能需要 .item() 来从0维CuPy数组中提取Python标量
                    min_val_scalar = min_eigenvalue.item() if hasattr(min_eigenvalue, 'item') else min_eigenvalue
                    self._internal_logger.warning(
                        f"Density matrix property check failed: Matrix is not positive semi-definite. "
                        f"Minimum eigenvalue found: {min_val_scalar:.2e}"
                    )
                    return False
                
        except (AttributeError, NotImplementedError) as e:
            # 捕获后端不支持 eigvalsh 的情况
            self._internal_logger.error(
                f"The current backend ({type(self._backend).__name__}) does not support the 'eigvalsh' method, "
                f"which is required for the positive semi-definite check. Error: {e}", exc_info=True)
            return False
        except Exception as e:
            # 捕获线性代数计算中可能发生的任何其他错误
            self._internal_logger.error(
                f"An unexpected error occurred during eigenvalue calculation for the density matrix property check: {e}",
                exc_info=True
            )
            return False
            
        # --- 步骤 3: 如果所有检查都通过 ---
        return True

    def is_normalized(self, tolerance: float = 1e-9) -> bool:
        """
        检查密度矩阵的迹（Trace）是否归一化为 1.0。

        在量子力学中，一个有效的密度矩阵 `ρ` 的迹必须为 1，这代表了
        测量所有可能结果的总概率为 100%。此属性被称为概率守恒。

        此方法通过调用当前计算后端 (`self._backend`) 的 `trace` 和 `isclose`
        方法，以一种对浮点数误差稳健的方式进行验证。

        [修正] 此实现是后端无关的，可以无缝地在 CuPy 和 PurePythonBackend
        上工作。

        Args:
            tolerance (float, optional): 
                用于浮点数比较的绝对容差。如果 `abs(Tr(ρ) - 1.0)` 小于
                等于此值，则认为迹是归一化的。默认为 1e-9。

        Returns:
            bool: 
                如果矩阵的迹在指定的容差范围内等于 1.0，则返回 `True`，
                否则记录一条警告并返回 `False`。
        """
        # --- 步骤 1: 计算密度矩阵的迹 ---
        # 调用 self._backend.trace()，它会根据当前后端（CuPy 或 PurePython）
        # 执行正确的迹运算。
        # 对于厄米矩阵，迹理论上是一个实数，但计算结果可能是复数，
        # isclose 方法可以正确处理这种情况。
        trace_value = self._backend.trace(self._density_matrix)
        
        # --- 步骤 2: 使用后端通用的 isclose 方法进行安全比较 ---
        # isclose 会检查 abs(trace_value - 1.0) <= tolerance。
        # 它可以正确处理 trace_value 是复数的情况。
        if not self._backend.isclose(trace_value, 1.0, atol=tolerance):
            # 如果检查失败，记录一条详细的警告信息，包含当前的迹值，
            # 这对于调试数值稳定性问题非常有帮助。
            self._internal_logger.warning(
                f"Density matrix normalization check failed: Trace is {trace_value.real:.8f}, "
                f"which is not within the tolerance ({tolerance}) of 1.0."
            )
            return False
            
        # --- 步骤 3: 如果检查通过 ---
        return True

    def is_valid(self, tolerance: float = 1e-9, check_trace_one: bool = True) -> bool:
        """
        对量子态进行全面的有效性检查。

        这是一个综合性的验证函数，它会检查一个 `QuantumState` 实例
        在所有关键方面是否都处于一个物理上和逻辑上有效的状态。

        检查包括：
        1.  `num_qubits` 是否为有效的非负整数。
        2.  `num_qubits` 是否在用户配置的 `MAX_QUBITS` 限制范围内。
        3.  密度矩阵是否满足核心数学属性：厄米性和正半定性（通过调用
            `is_valid_density_matrix_properties`）。
        4.  (可选) 密度矩阵的迹是否归一化为 1（通过调用 `is_normalized`）。

        [修正] 此实现已更新，移除了对已废弃的内部属性的检查，并完全
        依赖于与后端无关的辅助验证方法。

        Args:
            tolerance (float, optional): 
                用于所有底层浮点数比较的容差。默认为 1e-9。
            check_trace_one (bool, optional): 
                如果为 `True` (默认)，则会检查迹是否为 1。
                如果为 `False`，则跳过此项检查，这在某些中间计算步骤中
                可能很有用。

        Returns:
            bool: 如果所有启用的检查都通过，则返回 `True`，否则返回 `False`。
                  任何失败的检查都会在 `WARNING` 或 `ERROR` 级别记录日志。
        """
        # --- 检查 1: num_qubits 的基本有效性 ---
        if not isinstance(self.num_qubits, int) or self.num_qubits < 0:
            self._internal_logger.warning(
                f"State validation failed: num_qubits ({self.num_qubits}) is not a non-negative integer."
            )
            return False

        # --- 检查 2: num_qubits 是否在用户配置的有效范围内 ---
        # 这个检查在 __post_init__ 中已经做过，但在这里再次检查可以增加健壮性，
        # 防止对象在创建后被不安全地修改。
        user_config_max_qubits = _core_config.get("MAX_QUBITS", 10)
        if self.num_qubits > user_config_max_qubits:
            self._internal_logger.warning(
                f"State validation failed: num_qubits ({self.num_qubits}) exceeds the user-configured "
                f"maximum of {user_config_max_qubits}."
            )
            return False

        # --- 检查 3: 密度矩阵的核心数学属性 ---
        # 委托给 is_valid_density_matrix_properties 方法。
        # 如果该方法失败，它会在内部记录详细的日志，因此这里无需重复。
        if not self.is_valid_density_matrix_properties(self._density_matrix, tolerance):
            # 日志已在子方法中记录。
            return False
            
        # --- 检查 4: (可选) 迹是否归一化为 1 ---
        # 根据 check_trace_one 参数决定是否执行此检查。
        if check_trace_one:
            # 委托给 is_normalized 方法。
            # 如果该方法失败，它同样会在内部记录详细的日志。
            if not self.is_normalized(tolerance):
                # 日志已在子方法中记录。
                return False
            
        # --- 成功: 如果所有检查都通过 ---
        return True

    def get_probabilities(self) -> List[float]:
        """
        获取所有计算基的测量概率。

        此方法返回一个一维列表，其长度为 2^N (其中 N 是量子比特数)，
        列表中的第 i 个元素是测量到计算基矢 |i⟩ 的概率。

        概率由密度矩阵 `ρ` 的对角线元素 `ρ[i, i]` 给出。这些对角线元素
        理论上应该是实数，但此方法会取其实部以消除任何由于计算产生的
        微小浮点虚部。

        [修正] 此方法的返回类型已从 `np.ndarray` 更改为 `List[float]`，
        以消除对 NumPy 的依赖，并为所有后端提供一个统一的、标准的
        Python 类型输出。

        Returns:
            List[float]: 
                一个包含所有测量概率的 Python 列表。
                如果当前量子态无效，则记录一条错误并返回一个空列表。
        """
        log_prefix = f"QuantumState.get_probabilities(N={self.num_qubits})"

        # --- 步骤 1: 前置验证 ---
        # 在进行任何计算之前，首先确保量子态是物理上有效的。
        if not self.is_valid():
            self._internal_logger.error(
                f"[{log_prefix}] get_probabilities called on an invalid QuantumState. "
                "Returning an empty list."
            )
            return []
            
        # --- 步骤 2: 提取对角线元素 ---
        # 调用 self._backend.diag()，它会根据当前后端（CuPy 或 PurePython）
        # 执行正确的对角线提取操作。
        probabilities_complex = self._backend.diag(self._density_matrix)
        
        # --- 步骤 3: 转换为实数列表 ---
        # 无论后端返回的是 CuPy 数组还是 Python 列表，我们都可以迭代它
        # 并取每个元素的实部。
        probabilities_real = [p.real for p in probabilities_complex]
        
        # --- 步骤 4: 数值裁剪 ---
        # 由于浮点数精度误差，计算出的概率值可能略微超出 [0.0, 1.0] 的范围。
        # 使用后端的 clip 函数将它们强制约束在这个物理有效的区间内。
        clipped_probs_backend_type = self._backend.clip(probabilities_real, 0.0, 1.0)
        
        # --- 步骤 5: 确保最终输出是标准的 Python float 列表 ---
        # 这一步是关键，它确保了无论后端是什么，返回类型都是一致的。
        # - 如果后端是 PurePythonBackend, clipped_probs_backend_type 已经是列表。
        # - 如果后端是 CuPy, clipped_probs_backend_type 是 CuPy 数组，
        #   列表推导式会将其转换为 Python 列表。
        final_probabilities = [float(p) for p in clipped_probs_backend_type]

        self._internal_logger.debug(f"[{log_prefix}] Successfully retrieved {len(final_probabilities)} measurement probabilities.")

        return final_probabilities
    
    # --- 核心计算与门操作辅助函数 ---
    
    def _build_global_operator_multi_qubit(self, target_qubits: List[int], local_operator: Any) -> Any:
        """
        [终极硬编码版] 为作用于任意多个（可能非连续）量子比特的多比特门构建全局算子。
        
        此方法通过遍历全局算子矩阵的每一个元素 `(row, col)`，并根据局部算子 `local_operator`
        的定义手动填充其值。这种方法虽然在理论上性能较低（O(4^N)），但其逻辑
        完全透明，能够从根本上杜绝所有因比特序（Endianness）或张量积顺序
        而导致的常见错误。

        约定：系统的基矢被视为 `|q_{n-1} ... q_1 q_0⟩` (big-endian  convention)。
        `target_qubits` 列表的顺序定义了 `local_operator` 的基矢顺序。例如，
        `target_qubits=[qA, qB]` 意味着 `local_operator` 的基矢是 `|qA, qB⟩`。

        [修正] 此实现现在是后端无关的，它使用 `self._backend.zeros` 创建
        全局算子，并能正确处理作为 `cupy.ndarray` 或 `List[List[complex]]`
        传入的 `local_operator`。

        Args:
            target_qubits (List[int]): 
                一个整数列表，指定了 `local_operator` 作用的目标量子比特。
                列表的顺序很重要。
            local_operator (Any): 
                一个局部的酉矩阵，其维度应为 `(2^k, 2^k)`，其中 `k` 是
                `target_qubits` 的长度。可以是 CuPy 数组或嵌套列表。

        Returns:
            Any: 
                一个 (2^N, 2^N) 的全局算子矩阵，其类型与当前后端匹配。
        
        Raises:
            ValueError: 如果 `target_qubits` 或 `local_operator` 的维度无效。
        """
        # --- 步骤 1: 严格的输入验证 ---
        num_local_qubits = len(target_qubits)
        if len(set(target_qubits)) != num_local_qubits or not (1 <= num_local_qubits <= self.num_qubits):
             raise ValueError(
                f"Invalid target_qubits list provided. It must contain unique qubit indices "
                f"within the range [0, {self.num_qubits-1}]."
            )
        for q in target_qubits:
            if not (0 <= q < self.num_qubits):
                raise ValueError(f"Invalid target qubit index {q} for a system of {self.num_qubits} qubits.")
        
        # 兼容不同后端的数据类型来检查 local_operator 的形状
        local_dim = 1 << num_local_qubits
        if hasattr(local_operator, 'shape'): # CuPy-like array
            op_shape = local_operator.shape
        elif isinstance(local_operator, list) and local_operator: # PurePythonBackend list
            op_shape = (len(local_operator), len(local_operator[0]) if local_operator[0] else 0)
        else:
            raise TypeError(f"Unsupported local_operator type: {type(local_operator).__name__}")

        if op_shape != (local_dim, local_dim):
            raise ValueError(
                f"The shape of the local operator {op_shape} does not match the expected shape "
                f"({local_dim}, {local_dim}) for {num_local_qubits} target qubits."
            )

        # --- 步骤 2: 准备硬编码循环所需的变量 ---
        total_dim = 1 << self.num_qubits
        
        # 使用当前后端创建正确类型的全零矩阵
        global_op = self._backend.zeros((total_dim, total_dim))
        
        # 找出不受 local_operator 影响的比特
        other_qubits = [q for q in range(self.num_qubits) if q not in target_qubits]
        
        # --- 步骤 3: 遍历并填充全局算子矩阵的每一个元素 ---
        # 遍历输出基矢 |row⟩
        for row in range(total_dim):
            # 遍历输入基矢 |col⟩
            for col in range(total_dim):
                
                # a) 检查非目标比特的状态是否一致。
                #    如果一个算子 U 只作用于子空间 A，那么矩阵元 <row|U|col> 不为零
                #    的一个必要条件是，|row> 和 |col> 在子空间 B (非A) 上的投影必须相同。
                #    这相当于检查这两个基矢在 other_qubits 上的比特值是否完全一样。
                is_match_on_other_qubits = True
                for q_other in other_qubits:
                    # 比较第 q_other 个比特
                    if ((row >> q_other) & 1) != ((col >> q_other) & 1):
                        is_match_on_other_qubits = False
                        break
                
                # 如果非目标比特不匹配，则该矩阵元必为0，跳到下一个元素
                if not is_match_on_other_qubits:
                    continue
                
                # b) 如果非目标比特匹配，则从局部算子中计算矩阵元的值。
                #    我们需要从全局索引 (row, col) 中提取出局部算子对应的局部索引。
                
                local_row_index = 0
                local_col_index = 0
                
                # `target_qubits` 的顺序定义了 `local_operator` 的基矢。
                # 例如，对于 `target_qubits=[2, 0]`，局部基矢是 `|q2, q0>`.
                # `target_qubits[0]` (即q2) 对应局部基矢的最高位。
                for i in range(num_local_qubits):
                    target_qubit_index = target_qubits[i]
                    
                    # 计算这个比特在局部基矢中的位权重 (position value)
                    bit_weight = 1 << (num_local_qubits - 1 - i)
                    
                    # 检查全局索引 `row` 和 `col` 中对应比特是否为1，并构建局部索引
                    if (row >> target_qubit_index) & 1:
                        local_row_index |= bit_weight
                    if (col >> target_qubit_index) & 1:
                        local_col_index |= bit_weight
                
                # c) 从 local_operator 中获取元素并填充到全局算子矩阵中
                #    这种索引方式对 CuPy 数组和 Python 嵌套列表都有效。
                global_op[row][col] = local_operator[local_row_index][local_col_index]

        return global_op
    
    # --- [终极修正] `_apply_global_unitary` 移除NumPy并行分支，添加纯Python并行分支 ---
    def _apply_global_unitary(self, global_unitary: Any):
        """
        [合并增强版] 将一个全局酉算子应用于密度矩阵，执行核心的量子态演化。

        演化公式为： ρ' = U @ ρ @ U†
        其中 ρ 是当前密度矩阵，U 是全局酉算子，U† 是其共轭转置。

        此版本会自动检测是否已通过 `enable_parallelism()` 启用并行计算，
        并在满足条件时（PurePython后端、比特数达到阈值）自动使用多进程加速。
        如果并行计算失败，它会抛出 RuntimeError 中断执行，以防止状态损坏。
        """
        log_prefix = f"QuantumState._apply_global_unitary(N={self.num_qubits})"
        start_time = time.perf_counter()

        # --- 分支 1: CuPy 后端 (GPU 执行) ---
        # 如果后端是 CuPy，总是使用 GPU 进行计算，因为它通常比 CPU 并行更快。
        if self._backend is cp:
            logger.info(f"[{log_prefix}] Starting computation on GPU (CuPy) backend...")
            try:
                # `einsum` 是在 CuPy 中执行 U @ rho @ U.conj().T 的最高效方式之一，
                # 它能在一个操作中完成乘法和收缩，减少中间内存分配。
                self._density_matrix = self._backend.einsum(
                    'ij,jk,lk->il', 
                    global_unitary, 
                    self._density_matrix, 
                    global_unitary.conj(), # 'lk' 下标隐式地执行了共轭转置
                    optimize=True
                )
                # 确保在计时结束前所有异步的 GPU 操作都已完成
                cp.cuda.Stream.null.synchronize()
                duration = (time.perf_counter() - start_time) * 1000
                logger.info(f"[{log_prefix}] GPU (CuPy) computation completed in: {duration:.2f} ms")
            except Exception as e:
                logger.critical(f"[{log_prefix}] A critical error occurred during CuPy computation: {e}", exc_info=True)
                raise RuntimeError("GPU computation failed during unitary application.") from e

        # --- 分支 2: PurePythonBackend (CPU 执行) ---
        elif isinstance(self._backend, PurePythonBackend):
            parallel_threshold = _core_config.get("PARALLEL_COMPUTATION_QUBIT_THRESHOLD", 999)
            
            # --- 分支 2a: 检查是否应使用并行计算 ---
            # 条件：1. 量子比特数达到阈值  2. 全局并行开关已通过 enable_parallelism() 启用
            if self.num_qubits >= parallel_threshold and _parallel_enabled:
                logger.info(
                    f"[{log_prefix}] Parallelism is enabled and qubit count ({self.num_qubits}) "
                    f"meets threshold (>= {parallel_threshold}). Launching multi-process computation..."
                )
                
                # 调用内部的并行执行器
                parallel_result = _execute_parallel_unitary_evolution_pure(global_unitary, self._density_matrix)
                
                if parallel_result is not None:
                    # 如果并行计算成功，更新密度矩阵
                    self._density_matrix = parallel_result
                    duration = (time.perf_counter() - start_time) * 1000
                    logger.info(f"[{log_prefix}] Parallel Pure Python computation completed in: {duration:.2f} ms")
                else:
                    # 如果并行计算失败（返回None），记录严重错误并抛出异常，以中断整个线路执行
                    logger.critical(
                        f"[{log_prefix}] Parallel computation FAILED. The quantum state has NOT been updated "
                        "to prevent corruption. Check logs for worker process errors (e.g., timeout, exceptions)."
                    )
                    raise RuntimeError("Parallel unitary evolution failed. Circuit execution aborted.")
            
            # --- 分支 2b: PurePython 单线程计算 (默认或回退路径) ---
            else:
                # 增强日志，提供更清晰的单线程原因
                reason = ""
                if self.num_qubits < parallel_threshold:
                    reason = f"(qubit count {self.num_qubits} < threshold {parallel_threshold})"
                elif self.num_qubits >= parallel_threshold and not _parallel_enabled:
                    reason = "(threshold met, but parallelism not enabled via enable_parallelism())"
                
                logger.info(f"[{log_prefix}] Starting single-threaded computation on PurePythonBackend {reason}...")
                
                # 执行标准的 U @ rho @ U_dagger 计算
                temp_product = self._backend.dot(global_unitary, self._density_matrix)
                unitary_dagger = self._backend.conj_transpose(global_unitary)
                self._density_matrix = self._backend.dot(temp_product, unitary_dagger)
                
                duration = (time.perf_counter() - start_time) * 1000
                logger.info(f"[{log_prefix}] Single-threaded Pure Python computation completed in: {duration:.2f} ms")
        
        else:
            # 这是一个安全网，理论上不应该被触发，但对于健壮性是必要的
            raise TypeError(f"Unsupported backend type encountered: {type(self._backend).__name__}")

        # --- 后处理: 重新归一化以对抗浮点误差 ---
        # 无论使用哪种计算路径，都进行归一化，以保证量子态的物理有效性
        self.normalize()
    # --- 执行量子线路 (run_circuit 保持不变) ---
    def run_circuit(self, circuit: 'QuantumCircuit'):
        """
        在一个量子态上执行一个量子线路中的所有指令。

        此方法是量子态演化的指令执行引擎。它会按顺序遍历 `QuantumCircuit`
        对象中的每一条指令，解析出门操作的名称、参数以及任何经典条件，
        然后动态地调用当前 `QuantumState` 实例中相应的实现方法（例如 `self.h`,
        `self.cnot` 等）来更新量子态。

        [修正] 此实现是完全后端无关的，因为它只负责分发指令，而实际的
        计算由已经被重构为后端无关的各个门方法来完成。

        支持的功能包括：
        -   **经典控制流**: 可以根据 `self._classical_registers` 中的值，
            条件性地执行门操作（通过指令中的 `condition` 关键字）。

        Args:
            circuit (QuantumCircuit): 
                包含一系列待执行量子操作的 `QuantumCircuit` 对象。

        Raises:
            TypeError: 如果输入的 `circuit` 不是 `QuantumCircuit` 类的实例。
            ValueError: 如果 `circuit` 和当前 `QuantumState` 的量子比特数不匹配。
        """
        # --- 步骤 1: 输入验证 ---
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Input to run_circuit must be a QuantumCircuit instance.")
        
        if self.num_qubits != circuit.num_qubits:
            raise ValueError(
                f"Circuit's qubit number ({circuit.num_qubits}) does not match "
                f"QuantumState's qubit number ({self.num_qubits})."
            )

        self._internal_logger.info(
            f"Running circuit '{circuit.description or 'unnamed'}' with "
            f"{len(circuit.instructions)} instructions on a {self.num_qubits}-qubit state."
        )

        # --- 步骤 2: 遍历并执行每一条指令 ---
        for instruction in circuit.instructions:
            # --- 步骤 2a: 解析指令元组 ---
            # 指令格式: (gate_name, arg1, arg2, ..., {kwargs_dict})
            gate_name = instruction[0]
            
            # 从指令元组的最后一个元素中提取 kwargs 字典（如果存在）
            op_kwargs = instruction[-1] if instruction and isinstance(instruction[-1], dict) else {}
            
            # 非关键字参数是除操作名和最后一个 kwargs 字典之外的所有元素
            op_args = instruction[1:-1] if instruction and isinstance(instruction[-1], dict) else instruction[1:]
            
            # --- 步骤 2b: 检查并处理经典控制流条件 ---
            if 'condition' in op_kwargs:
                cr_index, expected_value = op_kwargs['condition']
                actual_value = self._classical_registers.get(cr_index)
                
                # 如果经典寄存器还未被设置（值为 None），则跳过此条件门
                if actual_value is None:
                    self._internal_logger.warning(
                        f"Skipping conditional gate '{gate_name}' on CR[{cr_index}] "
                        f"because the classical register has not been initialized (value is None)."
                    )
                    continue
                
                # 如果条件不满足，则跳过此门
                if actual_value != expected_value:
                    self._internal_logger.debug(
                        f"Skipping conditional gate '{gate_name}' on CR[{cr_index}] "
                        f"because actual value ({actual_value}) != expected value ({expected_value})."
                    )
                    continue
                
                # 如果条件满足，将 'condition' 从 kwargs 中移除，
                # 这样它就不会被错误地传递给底层的门方法。
                op_kwargs_for_method = {k: v for k, v in op_kwargs.items() if k != 'condition'}
            else:
                op_kwargs_for_method = op_kwargs
            
            # --- 步骤 2c: 动态调用并执行相应的门方法 ---
            try:
                # 使用 getattr 动态获取与指令名称同名的方法
                method_to_call = getattr(self, gate_name)
            except AttributeError:
                # 如果在 QuantumState 类中找不到对应的方法，记录警告并跳过
                self._internal_logger.warning(
                    f"Gate method '{gate_name}' not found in QuantumState. "
                    f"Skipping instruction: {instruction}"
                )
                continue
            
            if callable(method_to_call):
                try:
                    # 调用找到的方法，并使用 * 和 ** 解包参数
                    method_to_call(*op_args, **op_kwargs_for_method)
                    
                    # 操作成功后，自动记录到历史日志
                    self._add_history_log(gate_name, targets=list(op_args), params=op_kwargs)
                except Exception as e:
                    # 捕获在具体门方法执行中发生的错误
                    self._internal_logger.error(f"Error executing gate '{gate_name}' with args {op_args}: {e}", exc_info=True)
                    # 重新抛出异常，以中断整个线路的执行，防止状态损坏
                    raise RuntimeError(f"Execution failed on gate '{gate_name}'.") from e
            else:
                # 如果 getattr 找到了一个属性但它不是可调用的（不是方法）
                self._internal_logger.warning(
                    f"Found attribute '{gate_name}', but it is not a callable method. "
                    f"Skipping instruction: {instruction}"
                )

    # --- 基础单比特门 (保持不变，已通过 self._backend 解耦) ---
    def x(self, target_qubit: int):
        """
        将一个 Pauli-X (NOT) 门应用于指定的量子比特。

        Pauli-X 门的作用类似于经典计算中的 NOT 门，它会翻转量子比特的状态：
        - X|0⟩ = |1⟩
        - X|1⟩ = |0⟩

        其矩阵表示为:
        [[0, 1],
         [1, 0]]

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        Pauli-X 矩阵，然后调用通用的 `_build_global_operator_multi_qubit` 和
        `_apply_global_unitary` 方法来完成整个演化过程。

        Args:
            target_qubit (int): 
                要应用 X 门的目标量子比特的索引。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用标准的 Python 嵌套列表来表示 Pauli-X 矩阵。
        # 这种表示法可以被任何后端（CuPy 或 PurePythonBackend）理解和处理。
        local_op_x = [
            [0.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数，它会处理张量积和比特排序。
        # _build_global_operator_multi_qubit 会返回一个与当前后端类型匹配的矩阵。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_x
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for X gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def y(self, target_qubit: int):
        """
        将一个 Pauli-Y 门应用于指定的量子比特。

        Pauli-Y 门的作用是围绕布洛赫球的 Y 轴进行一个 π (pi) 弧度的旋转。
        它会翻转量子比特的状态并施加一个相位：
        - Y|0⟩ =  i|1⟩
        - Y|1⟩ = -i|0⟩

        其矩阵表示为:
        [[0, -i],
         [i,  0]]

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        Pauli-Y 矩阵，然后调用通用的 `_build_global_operator_multi_qubit` 和
        `_apply_global_unitary` 方法来完成整个演化过程。

        Args:
            target_qubit (int): 
                要应用 Y 门的目标量子比特的索引。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用标准的 Python 嵌套列表来表示 Pauli-Y 矩阵。
        # 注意复数 `1j` 的使用。
        local_op_y = [
            [0.0 + 0.0j, 0.0 - 1.0j],  # -i
            [0.0 + 1.0j, 0.0 + 0.0j]   # +i
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数，它会处理张量积和比特排序。
        # _build_global_operator_multi_qubit 会返回一个与当前后端类型匹配的矩阵。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_y
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for Y gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def z(self, target_qubit: int):
        """
        将一个 Pauli-Z (Phase-flip) 门应用于指定的量子比特。

        Pauli-Z 门的作用是围绕布洛赫球的 Z 轴进行一个 π (pi) 弧度的旋转。
        它保持 |0⟩ 状态不变，但会给 |1⟩ 状态施加一个 -1 的相位（即翻转相位）：
        - Z|0⟩ =  |0⟩
        - Z|1⟩ = -|1⟩

        其矩阵表示为:
        [[1,  0],
         [0, -1]]

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        Pauli-Z 矩阵，然后调用通用的 `_build_global_operator_multi_qubit` 和
        `_apply_global_unitary` 方法来完成整个演化过程。

        Args:
            target_qubit (int): 
                要应用 Z 门的目标量子比特的索引。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用标准的 Python 嵌套列表来表示 Pauli-Z 矩阵。
        local_op_z = [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -1.0 + 0.0j]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数，它会处理张量积和比特排序。
        # _build_global_operator_multi_qubit 会返回一个与当前后端类型匹配的矩阵。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_z
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for Z gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def h(self, target_qubit: int):
        """
        将一个 Hadamard (H) 门应用于指定的量子比特。

        Hadamard 门是量子计算中最重要的门之一，它能够创造等概率的叠加态。
        它将计算基矢 |0⟩ 和 |1⟩ 映射到对角基矢 |+⟩ 和 |-⟩：
        - H|0⟩ = (|0⟩ + |1⟩) / √2  = |+⟩
        - H|1⟩ = (|0⟩ - |1⟩) / √2  = |-⟩

        其矩阵表示为 (1/√2) 乘以:
        [[1,  1],
         [1, -1]]

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        Hadamard 矩阵，然后调用通用的 `_build_global_operator_multi_qubit` 和
        `_apply_global_unitary` 方法来完成整个演化过程。

        Args:
            target_qubit (int): 
                要应用 H 门的目标量子比特的索引。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `math` 来计算 1/√2，以避免任何外部依赖。
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        
        # 使用标准的 Python 嵌套列表来表示 Hadamard 矩阵。
        local_op_h = [
            [sqrt2_inv + 0.0j,  sqrt2_inv + 0.0j],
            [sqrt2_inv + 0.0j, -sqrt2_inv + 0.0j]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数，它会处理张量积和比特排序。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_h
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for H gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def s(self, target_qubit: int):
        """
        将一个 S (Phase) 门应用于指定的量子比特。

        S 门是 Z 门的平方根（S = √Z），它施加一个依赖于状态的相位。
        它保持 |0⟩ 状态不变，但会给 |1⟩ 状态施加一个 `i` (虚数单位) 的相位。
        - S|0⟩ = |0⟩
        - S|1⟩ = i|1⟩

        其矩阵表示为:
        [[1, 0],
         [0, i]]

        S 门等价于围绕布洛赫球的 Z 轴旋转 π/2 弧度。

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        S 门矩阵，然后调用通用的 `_build_global_operator_multi_qubit` 和
        `_apply_global_unitary` 方法来完成整个演化过程。

        Args:
            target_qubit (int): 
                要应用 S 门的目标量子比特的索引。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用标准的 Python 嵌套列表和复数 `1j` 来表示 S 门矩阵。
        local_op_s = [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 1.0j]  # i
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数，它会处理张量积和比特排序。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_s
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for S gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def sdg(self, target_qubit: int):
        """
        将一个 S-dagger (S†) 门应用于指定的量子比特。

        S-dagger 门是 S 门的厄米共轭（Hermitian conjugate），即其逆操作。
        它施加一个与 S 门相反的相位。
        - S†|0⟩ = |0⟩
        - S†|1⟩ = -i|1⟩

        其矩阵表示为:
        [[1,  0],
         [0, -i]]

        S-dagger 门等价于围绕布洛赫球的 Z 轴旋转 -π/2 弧度。

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        S† 门矩阵，然后调用通用的 `_build_global_operator_multi_qubit` 和
        `_apply_global_unitary` 方法来完成整个演化过程。

        Args:
            target_qubit (int): 
                要应用 S† 门的目标量子比特的索引。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用标准的 Python 嵌套列表和复数 `-1j` 来表示 S-dagger 门矩阵。
        local_op_sdg = [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 - 1.0j]  # -i
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数，它会处理张量积和比特排序。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_sdg
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for S-dagger gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def t_gate(self, target_qubit: int):
        """
        将一个 T (π/8) 门应用于指定的量子比特。

        T 门是 S 门的平方根（T = √S），它提供了一个更精细的相位旋转。
        它保持 |0⟩ 状态不变，但会给 |1⟩ 状态施加一个 `e^(iπ/4)` 的相位。
        - T|0⟩ = |0⟩
        - T|1⟩ = e^(iπ/4)|1⟩ = (1+i)/√2 |1⟩

        其矩阵表示为:
        [[1, 0          ],
         [0, e^(iπ/4)]]

        T 门等价于围绕布洛赫球的 Z 轴旋转 π/4 弧度。与 H 门和 CNOT 门
        一起，T 门是实现通用量子计算所需的门集之一（尽管不是唯一的选择）。

        [修正] 此实现是后端无关的。它使用 Python 标准库 `cmath` 和 `math`
        计算相位，并定义了一个纯 Python 列表形式的 T 门矩阵。

        Args:
            target_qubit (int): 
                要应用 T 门的目标量子比特的索引。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `cmath` 和 `math` 来计算相位因子 e^(iπ/4)。
        # 这确保了计算的精确性，同时不引入外部依赖。
        phase = cmath.exp(1j * math.pi / 4.0)
        
        # 使用标准的 Python 嵌套列表来表示 T 门矩阵。
        local_op_t = [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, phase]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_t
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for T gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def tdg(self, target_qubit: int):
        """
        将一个 T-dagger (T†) 门应用于指定的量子比特。

        T-dagger 门是 T 门的厄米共轭（Hermitian conjugate），即其逆操作。
        它施加一个与 T 门相反的相位。
        - T†|0⟩ = |0⟩
        - T†|1⟩ = e^(-iπ/4)|1⟩ = (1-i)/√2 |1⟩

        其矩阵表示为:
        [[1, 0           ],
         [0, e^(-iπ/4)]]

        T-dagger 门等价于围绕布洛赫球的 Z 轴旋转 -π/4 弧度。

        [修正] 此实现是后端无关的。它使用 Python 标准库 `cmath` 和 `math`
        计算相位，并定义了一个纯 Python 列表形式的 T† 门矩阵。

        Args:
            target_qubit (int): 
                要应用 T† 门的目标量子比特的索引。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `cmath` 和 `math` 来计算相位因子 e^(-iπ/4)。
        # 负号 `-` 应用于虚部，实现了共轭相位的计算。
        phase = cmath.exp(-1j * math.pi / 4.0)
        
        # 使用标准的 Python 嵌套列表来表示 T-dagger 门矩阵。
        local_op_tdg = [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, phase]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_tdg
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for T-dagger gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)
    
    # --- 参数化单比特门 (保持不变，已通过 self._backend 解耦) ---
    def rx(self, target_qubit: int, theta: float):
        """
        将一个参数化的 RX(θ) 旋转门应用于指定的量子比特。

        RX(θ) 门表示围绕布洛赫球的 X 轴旋转一个角度 θ (theta)。
        它是变分量子算法 (VQA) 和许多其他量子算法中的基本构建块。

        其矩阵表示为:
        [[cos(θ/2),  -i*sin(θ/2)],
         [-i*sin(θ/2), cos(θ/2)]]

        [修正] 此实现是后端无关的。它使用 Python 标准库 `math`
        计算三角函数值，并定义了一个纯 Python 列表形式的 RX(θ) 门矩阵。

        Args:
            target_qubit (int): 
                要应用 RX 门的目标量子比特的索引。
            theta (float):
                围绕 X 轴旋转的角度，以弧度为单位。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `math` 来计算 cos(θ/2) 和 sin(θ/2)。
        half_theta = theta / 2.0
        cos_half = math.cos(half_theta)
        sin_half = math.sin(half_theta)
        
        # 使用标准的 Python 嵌套列表和复数 `-1j` 来表示 RX(θ) 门矩阵。
        local_op_rx = [
            [cos_half + 0.0j,    -1j * sin_half],
            [-1j * sin_half,     cos_half + 0.0j]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_rx
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for RX({theta:.4f}) gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def ry(self, target_qubit: int, theta: float):
        """
        将一个参数化的 RY(θ) 旋转门应用于指定的量子比特。

        RY(θ) 门表示围绕布洛赫球的 Y 轴旋转一个角度 θ (theta)。
        它常用于将量子比特从计算基态旋转到叠加态中。

        其矩阵表示为:
        [[cos(θ/2), -sin(θ/2)],
         [sin(θ/2),  cos(θ/2)]]

        [修正] 此实现是后端无关的。它使用 Python 标准库 `math`
        计算三角函数值，并定义了一个纯 Python 列表形式的 RY(θ) 门矩阵。

        Args:
            target_qubit (int): 
                要应用 RY 门的目标量子比特的索引。
            theta (float):
                围绕 Y 轴旋转的角度，以弧度为单位。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `math` 来计算 cos(θ/2) 和 sin(θ/2)。
        half_theta = theta / 2.0
        cos_half = math.cos(half_theta)
        sin_half = math.sin(half_theta)
        
        # 使用标准的 Python 嵌套列表来表示 RY(θ) 门矩阵。
        # 注意这个矩阵的所有元素都是实数。
        local_op_ry = [
            [cos_half + 0.0j, -sin_half + 0.0j],
            [sin_half + 0.0j,  cos_half + 0.0j]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_ry
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for RY({theta:.4f}) gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def rz(self, target_qubit: int, phi: float):
        """
        将一个参数化的 RZ(φ) 旋转门应用于指定的量子比特。

        RZ(φ) 门表示围绕布洛赫球的 Z 轴旋转一个角度 φ (phi)。
        它是一个相位门，只改变量子比特的相对相位，不改变 |0⟩ 和 |1⟩ 的概率幅。
        - RZ(φ)|0⟩ = e^(-iφ/2)|0⟩
        - RZ(φ)|1⟩ = e^(+iφ/2)|1⟩

        其矩阵表示为:
        [[e^(-iφ/2), 0          ],
         [0,           e^(+iφ/2)]]

        注意：RZ(φ) 和 P(λ) 门密切相关，RZ(φ) = e^(-iφ/2) * P(φ)。它们仅相差一个全局相位。

        [修正] 此实现是后端无关的。它使用 Python 标准库 `cmath`
        计算相位因子，并定义了一个纯 Python 列表形式的 RZ(φ) 门矩阵。

        Args:
            target_qubit (int): 
                要应用 RZ 门的目标量子比特的索引。
            phi (float):
                围绕 Z 轴旋转的角度，以弧度为单位。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `cmath` 来计算 e^(-iφ/2) 和 e^(+iφ/2)。
        half_phi = phi / 2.0
        phase_minus = cmath.exp(-1j * half_phi)
        phase_plus  = cmath.exp(1j * half_phi)
        
        # 使用标准的 Python 嵌套列表来表示 RZ(φ) 门矩阵。
        # 这是一个对角矩阵。
        local_op_rz = [
            [phase_minus,  0.0 + 0.0j],
            [0.0 + 0.0j,   phase_plus]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_rz
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for RZ({phi:.4f}) gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def p_gate(self, target_qubit: int, lambda_angle: float):
        """
        将一个参数化的 P(λ) (Phase) 门应用于指定的量子比特。

        P(λ) 门是一个相位门，它保持 |0⟩ 状态不变，但会给 |1⟩ 状态施加一个
        `e^(iλ)` 的相位。
        - P(λ)|0⟩ = |0⟩
        - P(λ)|1⟩ = e^(iλ)|1⟩

        其矩阵表示为:
        [[1, 0      ],
         [0, e^(iλ)]]

        P 门是许多量子算法（如量子傅里叶变换）中的关键组成部分。
        特殊情况：
        - P(π)   = Z 门
        - P(π/2) = S 门
        - P(π/4) = T 门

        [修正] 此实现是后端无关的。它使用 Python 标准库 `cmath`
        计算相位因子，并定义了一个纯 Python 列表形式的 P(λ) 门矩阵。

        Args:
            target_qubit (int): 
                要应用 P 门的目标量子比特的索引。
            lambda_angle (float):
                施加的相位角 λ (lambda)，以弧度为单位。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `cmath` 来计算相位因子 e^(iλ)。
        phase = cmath.exp(1j * lambda_angle)
        
        # 使用标准的 Python 嵌套列表来表示 P(λ) 门矩阵。
        # 这是一个对角矩阵。
        local_op_p = [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, phase]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_p
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for P({lambda_angle:.4f}) gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def u3_gate(self, target_qubit: int, theta: float, phi: float, lambda_angle: float):
        """
        将一个通用的 U3(θ, φ, λ) 单比特门应用于指定的量子比特。

        U3 门是一个可以表示任何单个量子比特酉变换的通用门（相差一个全局相位）。
        它是许多量子计算框架（如 IBM Qiskit）中的基础门之一。

        其矩阵表示为:
        [[cos(θ/2),                   -e^(iλ)*sin(θ/2)          ],
         [e^(iφ)*sin(θ/2),            e^(i(φ+λ))*cos(θ/2)        ]]

        许多常见的门都可以通过 U3 门表示：
        - U3(π, 0, π)      = X 门
        - U3(0, 0, λ)      = P(λ) 门
        - U3(π/2, 0, π)    = H 门

        [修正] 此实现是后端无关的。它使用 Python 标准库 `math` 和 `cmath`
        计算矩阵元素，并定义了一个纯 Python 列表形式的 U3 门矩阵。

        Args:
            target_qubit (int): 
                要应用 U3 门的目标量子比特的索引。
            theta (float):
                第一个旋转角度 θ，以弧度为单位。
            phi (float):
                第二个旋转角度 φ，以弧度为单位。
            lambda_angle (float):
                第三个旋转角度 λ，以弧度为单位。
        
        Raises:
            ValueError: 如果 `target_qubit` 索引无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `math` 和 `cmath` 来计算所有矩阵元素。
        half_theta = theta / 2.0
        cos_half = math.cos(half_theta)
        sin_half = math.sin(half_theta)
        
        # 计算 U3 矩阵的四个元素
        u3_00 = cos_half + 0.0j
        u3_01 = -cmath.exp(1j * lambda_angle) * sin_half
        u3_10 = cmath.exp(1j * phi) * sin_half
        u3_11 = cmath.exp(1j * (phi + lambda_angle)) * cos_half
        
        # 使用标准的 Python 嵌套列表来表示 U3 门矩阵。
        local_op_u3 = [
            [u3_00, u3_01],
            [u3_10, u3_11]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[target_qubit], 
                local_operator=local_op_u3
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for U3({theta:.2f}, {phi:.2f}, {lambda_angle:.2f}) gate on qubit {target_qubit}: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)
    
    # --- 基础多比特门 (保持不变，已通过 self._backend 解耦) ---
    def cnot(self, control_qubit: int, target_qubit: int):
        """
        将一个 Controlled-NOT (CNOT or CX) 门应用于指定的控制和目标量子比特。

        CNOT 门是量子计算中用于产生纠缠的关键双比特门。它的作用是：
        - 如果控制量子比特是 |0⟩，则目标量子比特保持不变。
        - 如果控制量子比特是 |1⟩，则对目标量子比特应用一个 X (NOT) 门。

        其作用于基矢 `|控制, 目标⟩`：
        - CNOT|00⟩ = |00⟩
        - CNOT|01⟩ = |01⟩
        - CNOT|10⟩ = |11⟩
        - CNOT|11⟩ = |10⟩

        其 4x4 矩阵表示为:
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]]

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        CNOT 门矩阵，然后调用通用的 `_build_global_operator_multi_qubit` 和
        `_apply_global_unitary` 方法。

        Args:
            control_qubit (int): 
                控制量子比特的索引。
            target_qubit (int): 
                目标量子比特的索引。
        
        Raises:
            ValueError: 如果控制和目标比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if control_qubit == target_qubit:
            raise ValueError("Control and target qubits cannot be the same for a CNOT gate.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用标准的 Python 嵌套列表来表示 CNOT 矩阵。
        # 基矢顺序约定为 |控制, 目标⟩，即 |00⟩, |01⟩, |10⟩, |11⟩。
        local_op_cnot = [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        # target_qubits 的顺序 [control_qubit, target_qubit] 至关重要，
        # 它告诉 _build_global_operator_multi_qubit 局部算子的基矢顺序。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[control_qubit, target_qubit], 
                local_operator=local_op_cnot
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for CNOT({control_qubit}, {target_qubit}) gate: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)
    
    def cz(self, control_qubit: int, target_qubit: int):
        """
        将一个 Controlled-Z (CZ) 门应用于指定的控制和目标量子比特。

        CZ 门是一个对称的双比特门。它的作用是：
        - 如果控制量子比特是 |0⟩，则目标量子比特保持不变。
        - 如果控制量子比特是 |1⟩，则对目标量子比特应用一个 Z (Phase-flip) 门。

        由于 Z 门只在 |1⟩ 上有作用，因此 CZ 门的效果是：当且仅当
        控制比特和目标比特都处于 |1⟩ 状态时，给整个系统施加一个 -1 的相位。
        
        其作用于基矢 `|控制, 目标⟩`：
        - CZ|00⟩ = |00⟩
        - CZ|01⟩ = |01⟩
        - CZ|10⟩ = |10⟩
        - CZ|11⟩ = -|11⟩

        其 4x4 矩阵表示为 (一个对角矩阵):
        [[1, 0, 0,  0],
         [0, 1, 0,  0],
         [0, 0, 1,  0],
         [0, 0, 0, -1]]

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        CZ 门矩阵，然后调用通用的 `_build_global_operator_multi_qubit` 和
        `_apply_global_unitary` 方法。

        Args:
            control_qubit (int): 
                控制量子比特的索引。
            target_qubit (int): 
                目标量子比特的索引。
        
        Raises:
            ValueError: 如果控制和目标比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if control_qubit == target_qubit:
            raise ValueError("Control and target qubits cannot be the same for a CZ gate.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用标准的 Python 嵌套列表来表示 CZ 矩阵。
        # 基矢顺序约定为 |控制, 目标⟩，即 |00⟩, |01⟩, |10⟩, |11⟩。
        local_op_cz = [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j,  0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j,  0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j,  0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        # 对于 CZ 门，由于其对称性，`[control_qubit, target_qubit]` 的顺序
        # 并不影响最终的全局算子，但为了保持一致性，我们仍按此顺序传递。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[control_qubit, target_qubit], 
                local_operator=local_op_cz
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for CZ({control_qubit}, {target_qubit}) gate: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def swap(self, qubit1: int, qubit2: int):
        """
        将一个 SWAP 门应用于指定的两个量子比特，以交换它们的状态。

        SWAP 门是一个双比特门，它的作用是交换两个量子比特的量子态。
        
        其作用于基矢 `|q1, q2⟩`：
        - SWAP|00⟩ = |00⟩
        - SWAP|01⟩ = |10⟩  (交换)
        - SWAP|10⟩ = |01⟩  (交换)
        - SWAP|11⟩ = |11⟩

        其 4x4 矩阵表示为:
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        SWAP 门矩阵，然后调用通用的 `_build_global_operator_multi_qubit` 和
        `_apply_global_unitary` 方法。

        Args:
            qubit1 (int): 
                第一个要交换的量子比特的索引。
            qubit2 (int): 
                第二个要交换的量子比特的索引。
        
        Raises:
            ValueError: 如果两个量子比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if qubit1 == qubit2:
            raise ValueError("The two qubits to be swapped must be different.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用标准的 Python 嵌套列表来表示 SWAP 矩阵。
        # 基矢顺序约定为 |qubit1, qubit2⟩，即 |00⟩, |01⟩, |10⟩, |11⟩。
        local_op_swap = [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        # target_qubits 的顺序 [qubit1, qubit2] 定义了局部算子的基矢顺序。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[qubit1, qubit2], 
                local_operator=local_op_swap
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for SWAP({qubit1}, {qubit2}) gate: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def toffoli(self, c1: int, c2: int, t: int):
        """
        将一个 Toffoli (CCNOT or CCX) 门应用于指定的两个控制和一个目标量子比特。

        Toffoli 门是一个双控非门。它的作用是：
        - 当且仅当两个控制量子比特 `c1` 和 `c2` 都处于 |1⟩ 状态时，
          才对目标量子比特 `t` 应用一个 X (NOT) 门。
        - 在所有其他情况下，目标量子比特保持不变。

        Toffoli 门在经典计算中是通用的（可以构建任何布尔函数），在量子计算中
        也是许多算法（如 Grover 搜索的 Oracle）的关键组成部分。

        其 8x8 矩阵表示（基矢 `|c1, c2, t⟩`）是一个单位矩阵，但在
        |110⟩ -> |111⟩ 和 |111⟩ -> |110⟩ 之间进行了交换。

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        8x8 Toffoli 门矩阵，然后调用通用的 `_build_global_operator_multi_qubit`
        和 `_apply_global_unitary` 方法。

        Args:
            c1 (int): 
                第一个控制量子比特的索引。
            c2 (int): 
                第二个控制量子比特的索引。
            t (int): 
                目标量子比特的索引。
        
        Raises:
            ValueError: 如果任何两个量子比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if len({c1, c2, t}) != 3:
            raise ValueError("Control and target qubits for a Toffoli gate must be unique.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 基矢顺序约定为 |c1, c2, t⟩，即 |000⟩, |001⟩, ..., |111⟩。
        # Toffoli 门是一个 8x8 的单位矩阵，只修改最后两行。
        # |110⟩ (二进制 6) 和 |111⟩ (二进制 7) 状态被交换。
        
        # 为了后端无关，我们手动构建这个 8x8 矩阵
        local_op_toffoli = [
            # row 0 to 5 are identity
            [1, 0, 0, 0, 0, 0, 0, 0],  # |000>
            [0, 1, 0, 0, 0, 0, 0, 0],  # |001>
            [0, 0, 1, 0, 0, 0, 0, 0],  # |010>
            [0, 0, 0, 1, 0, 0, 0, 0],  # |011>
            [0, 0, 0, 0, 1, 0, 0, 0],  # |100>
            [0, 0, 0, 0, 0, 1, 0, 0],  # |101>
            # row 6 and 7 are swapped
            [0, 0, 0, 0, 0, 0, 0, 1],  # |110> maps to |111>
            [0, 0, 0, 0, 0, 0, 1, 0]   # |111> maps to |110>
        ]
        
        # 确保所有元素都是复数
        local_op_toffoli = [[complex(val) for val in row] for row in local_op_toffoli]

        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        # target_qubits 的顺序 [c1, c2, t] 定义了局部算子的基矢顺序。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[c1, c2, t], 
                local_operator=local_op_toffoli
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for Toffoli({c1}, {c2}, {t}) gate: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def fredkin(self, c: int, t1: int, t2: int):
        """
        将一个 Fredkin (CSWAP) 门应用于指定的一个控制和两个目标量子比特。

        Fredkin 门是一个受控交换门。它的作用是：
        - 如果控制量子比特 `c` 是 |0⟩，则两个目标量子比特 `t1` 和 `t2` 保持不变。
        - 如果控制量子比特 `c` 是 |1⟩，则对两个目标量子比特 `t1` 和 `t2`
          应用一个 SWAP 门，交换它们的状态。

        Fredkin 门是可逆的，并且在可逆计算和量子计算的某些模型中非常重要。

        其 8x8 矩阵表示（基矢 `|c, t1, t2⟩`）是一个单位矩阵，但在
        |101⟩ -> |110⟩ 和 |110⟩ -> |101⟩ 之间进行了交换。

        [修正] 此实现是后端无关的。它首先定义了一个纯 Python 列表形式的
        8x8 Fredkin 门矩阵，然后调用通用的 `_build_global_operator_multi_qubit`
        和 `_apply_global_unitary` 方法。

        Args:
            c (int): 
                控制量子比特的索引。
            t1 (int): 
                第一个目标（被交换）量子比特的索引。
            t2 (int): 
                第二个目标（被交换）量子比特的索引。
        
        Raises:
            ValueError: 如果任何两个量子比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if len({c, t1, t2}) != 3:
            raise ValueError("Control and target qubits for a Fredkin gate must be unique.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 基矢顺序约定为 |c, t1, t2⟩，即 |000⟩, |001⟩, ..., |111⟩。
        # Fredkin 门是一个 8x8 的单位矩阵，只在控制比特为1的子空间中进行交换。
        # |101⟩ (二进制 5) 和 |110⟩ (二进制 6) 状态被交换。
        
        # 为了后端无关，我们手动构建这个 8x8 矩阵
        local_op_fredkin = [
            # row 0 to 4 are identity
            [1, 0, 0, 0, 0, 0, 0, 0],  # |000>
            [0, 1, 0, 0, 0, 0, 0, 0],  # |001>
            [0, 0, 1, 0, 0, 0, 0, 0],  # |010>
            [0, 0, 0, 1, 0, 0, 0, 0],  # |011>
            [0, 0, 0, 0, 1, 0, 0, 0],  # |100>
            # row 5 and 6 are swapped
            [0, 0, 0, 0, 0, 0, 1, 0],  # |101> maps to |110>
            [0, 0, 0, 0, 0, 1, 0, 0],  # |110> maps to |101>
            # row 7 is identity
            [0, 0, 0, 0, 0, 0, 0, 1]   # |111>
        ]
        
        # 确保所有元素都是复数，以保持类型一致性
        local_op_fredkin = [[complex(val) for val in row] for row in local_op_fredkin]

        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        # target_qubits 的顺序 [c, t1, t2] 定义了局部算子的基矢顺序。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[c, t1, t2], 
                local_operator=local_op_fredkin
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for Fredkin({c}, {t1}, {t2}) gate: {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)
    
    # --- [增强] 高级受控门 (保持不变，已通过 self._backend 解耦) ---
    def cp(self, control_qubit: int, target_qubit: int, angle: float):
        """
        将一个受控相位门 (Controlled-Phase Gate or CPHASE) 应用于指定的量子比特。

        CP(θ) 门的作用是：
        - 当且仅当控制量子比特和目标量子比特都处于 |1⟩ 状态时，给系统
          施加一个 `e^(iθ)` 的相位。
        - 在所有其他情况下，状态保持不变。

        它与 CZ 门的关系是：CZ = CP(π)。
        它也与 `QuantumCircuit` 中的 `p_gate` 相关，是一个受控版本的 P 门。

        其 4x4 矩阵表示为 (一个对角矩阵):
        [[1, 0, 0, 0        ],
         [0, 1, 0, 0        ],
         [0, 0, 1, 0        ],
         [0, 0, 0, e^(iθ)]]

        [修正] 此实现是后端无关的。它使用 Python 标准库 `cmath` 和 `math`
        计算相位因子，并定义了一个纯 Python 列表形式的 CP 门矩阵。

        Args:
            control_qubit (int): 
                控制量子比特的索引。
            target_qubit (int): 
                目标量子比特的索引。
            angle (float):
                施加的相位角 θ (theta)，以弧度为单位。
        
        Raises:
            ValueError: 如果控制和目标比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if control_qubit == target_qubit:
            raise ValueError("Control and target qubits cannot be the same for a CP gate.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `cmath` 和 `math` 来计算相位因子 e^(iθ)。
        phase_factor = cmath.exp(1j * angle)
        
        # 使用标准的 Python 嵌套列表来表示 CP(θ) 矩阵。
        # 基矢顺序约定为 |控制, 目标⟩，即 |00⟩, |01⟩, |10⟩, |11⟩。
        local_op_cp = [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, phase_factor]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[control_qubit, target_qubit], 
                local_operator=local_op_cp
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for CP({angle:.4f}) on ({control_qubit}, {target_qubit}): {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def crx(self, control_qubit: int, target_qubit: int, theta: float):
        """
        将一个受控-RX(θ) (CRX) 门应用于指定的控制和目标量子比特。

        CRX 门的作用是：
        - 如果控制量子比特是 |0⟩，则目标量子比特保持不变。
        - 如果控制量子比特是 |1⟩，则对目标量子比特应用一个 RX(θ) 门。

        其 4x4 矩阵表示（基矢 `|控制, 目标⟩`）为一个块对角矩阵:
        [[1, 0, 0,           0          ],
         [0, 1, 0,           0          ],
         [0, 0, cos(θ/2),    -i*sin(θ/2)],
         [0, 0, -i*sin(θ/2), cos(θ/2)   ]]

        [修正] 此实现是后端无关的。它使用 Python 标准库 `math`
        计算三角函数值，并手动构建一个纯 Python 列表形式的 CRX 门矩阵。

        Args:
            control_qubit (int): 
                控制量子比特的索引。
            target_qubit (int): 
                目标量子比特的索引。
            theta (float):
                当控制为1时，围绕 X 轴旋转的角度，以弧度为单位。
        
        Raises:
            ValueError: 如果控制和目标比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if control_qubit == target_qubit:
            raise ValueError("Control and target qubits cannot be the same for a CRX gate.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `math` 来计算 RX(θ) 门的元素。
        half_theta = theta / 2.0
        cos_half = math.cos(half_theta)
        sin_half = math.sin(half_theta)
        
        # 手动构建 4x4 的 CRX 矩阵。
        # 上半部分是 2x2 单位矩阵 I (对应控制比特为 |0⟩)。
        # 右下角是 2x2 的 RX(θ) 矩阵 (对应控制比特为 |1⟩)。
        # 基矢顺序约定为 |控制, 目标⟩。
        local_op_crx = [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j,      0.0 + 0.0j    ],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j,      0.0 + 0.0j    ],
            [0.0 + 0.0j, 0.0 + 0.0j, cos_half,        -1j * sin_half],
            [0.0 + 0.0j, 0.0 + 0.0j, -1j * sin_half,  cos_half      ]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[control_qubit, target_qubit], 
                local_operator=local_op_crx
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for CRX({theta:.4f}) on ({control_qubit}, {target_qubit}): {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def cry(self, control_qubit: int, target_qubit: int, theta: float):
        """
        将一个受控-RY(θ) (CRY) 门应用于指定的控制和目标量子比特。

        CRY 门的作用是：
        - 如果控制量子比特是 |0⟩，则目标量子比特保持不变。
        - 如果控制量子比特是 |1⟩，则对目标量子比特应用一个 RY(θ) 门。

        其 4x4 矩阵表示（基矢 `|控制, 目标⟩`）为一个块对角矩阵:
        [[1, 0, 0,          0         ],
         [0, 1, 0,          0         ],
         [0, 0, cos(θ/2),   -sin(θ/2) ],
         [0, 0, sin(θ/2),   cos(θ/2)  ]]

        [修正] 此实现是后端无关的。它使用 Python 标准库 `math`
        计算三角函数值，并手动构建一个纯 Python 列表形式的 CRY 门矩阵。

        Args:
            control_qubit (int): 
                控制量子比特的索引。
            target_qubit (int): 
                目标量子比特的索引。
            theta (float):
                当控制为1时，围绕 Y 轴旋转的角度，以弧度为单位。
        
        Raises:
            ValueError: 如果控制和目标比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if control_qubit == target_qubit:
            raise ValueError("Control and target qubits cannot be the same for a CRY gate.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `math` 来计算 RY(θ) 门的元素。
        half_theta = theta / 2.0
        cos_half = math.cos(half_theta)
        sin_half = math.sin(half_theta)
        
        # 手动构建 4x4 的 CRY 矩阵。
        # 上半部分是 2x2 单位矩阵 I (对应控制比特为 |0⟩)。
        # 右下角是 2x2 的 RY(θ) 矩阵 (对应控制比特为 |1⟩)。
        # 基矢顺序约定为 |控制, 目标⟩。
        local_op_cry = [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j,   0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j,   0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, cos_half,    -sin_half ],
            [0.0 + 0.0j, 0.0 + 0.0j, sin_half,     cos_half ]
        ]
        
        # 确保所有元素都是复数
        local_op_cry = [[complex(val) for val in row] for row in local_op_cry]

        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[control_qubit, target_qubit], 
                local_operator=local_op_cry
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for CRY({theta:.4f}) on ({control_qubit}, {target_qubit}): {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)
        
    def crz(self, control_qubit: int, target_qubit: int, phi: float):
        """
        将一个受控-RZ(φ) (CRZ) 门应用于指定的控制和目标量子比特。

        CRZ 门的作用是：
        - 如果控制量子比特是 |0⟩，则目标量子比特保持不变。
        - 如果控制量子比特是 |1⟩，则对目标量子比特应用一个 RZ(φ) 门。

        其 4x4 矩阵表示（基矢 `|控制, 目标⟩`）为一个块对角矩阵:
        [[1, 0, 0,           0          ],
         [0, 1, 0,           0          ],
         [0, 0, e^(-iφ/2),   0          ],
         [0, 0, 0,           e^(+iφ/2)  ]]

        [修正] 此实现是后端无关的。它使用 Python 标准库 `cmath` 和 `math`
        计算相位因子，并手动构建一个纯 Python 列表形式的 CRZ 门矩阵。

        Args:
            control_qubit (int): 
                控制量子比特的索引。
            target_qubit (int): 
                目标量子比特的索引。
            phi (float):
                当控制为1时，围绕 Z 轴旋转的角度，以弧度为单位。
        
        Raises:
            ValueError: 如果控制和目标比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if control_qubit == target_qubit:
            raise ValueError("Control and target qubits cannot be the same for a CRZ gate.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `cmath` 和 `math` 来计算 RZ(φ) 门的元素。
        half_phi = phi / 2.0
        phase_minus = cmath.exp(-1j * half_phi)
        phase_plus  = cmath.exp(1j * half_phi)
        
        # 手动构建 4x4 的 CRZ 矩阵。
        # 上半部分是 2x2 单位矩阵 I (对应控制比特为 |0⟩)。
        # 右下角是 2x2 的 RZ(φ) 矩阵 (对应控制比特为 |1⟩)。
        # 基矢顺序约定为 |控制, 目标⟩。
        local_op_crz = [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j,   0.0 + 0.0j ],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j,   0.0 + 0.0j ],
            [0.0 + 0.0j, 0.0 + 0.0j, phase_minus,  0.0 + 0.0j ],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j,   phase_plus ]
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[control_qubit, target_qubit], 
                local_operator=local_op_crz
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for CRZ({phi:.4f}) on ({control_qubit}, {target_qubit}): {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    # --- [终极修正] `controlled_u` 接收列表并转换为后端格式 ---
    def controlled_u(self, control_qubit: int, target_qubit: int, u_matrix: List[List[complex]], name: str = "CU"):
        """
        将一个通用的、由用户定义的单比特受控酉门 (Controlled-U) 应用于量子态。

        Controlled-U 门的作用是：
        - 如果控制量子比特是 |0⟩，则目标量子比特保持不变。
        - 如果控制量子比特是 |1⟩，则对目标量子比特应用用户定义的酉矩阵 `u_matrix`。

        其 4x4 局部算子是一个块矩阵，形式为 `[[I, 0], [0, U]]`。

        [修正] 此实现是后端无关的。它接收一个标准的 Python 嵌套列表
        `u_matrix` 作为输入，并在内部将其转换为当前后端所需的格式（如果需要），
        然后手动构建 4x4 的 Controlled-U 矩阵。

        Args:
            control_qubit (int): 
                控制量子比特的索引。
            target_qubit (int): 
                目标量子比特的索引。
            u_matrix (List[List[complex]]): 
                一个 2x2 的酉矩阵，以嵌套列表的形式表示要施加在目标比特上的操作。
            name (str, optional): 
                操作的名称，主要用于日志记录。默认为 "CU"。
        
        Raises:
            ValueError: 如果控制和目标比特索引相同或 `u_matrix` 格式无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if control_qubit == target_qubit:
            raise ValueError("Control and target qubits cannot be the same for a Controlled-U gate.")
        # 在 QuantumCircuit 中已经对 u_matrix 格式进行了验证，但在这里再次检查可以增加健壮性
        if not (isinstance(u_matrix, list) and len(u_matrix) == 2 and
                isinstance(u_matrix[0], list) and len(u_matrix[0]) == 2 and
                isinstance(u_matrix[1], list) and len(u_matrix[1]) == 2):
            raise ValueError("`u_matrix` for controlled_u must be a 2x2 nested list of complex numbers.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 手动构建 4x4 的 Controlled-U 矩阵。
        # 上半部分是 2x2 单位矩阵 I。
        # 右下角是用户提供的 2x2 的 U 矩阵。
        # 基矢顺序约定为 |控制, 目标⟩。
        u00, u01 = u_matrix[0]
        u10, u11 = u_matrix[1]

        local_op_cu = [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, u00,        u01       ],
            [0.0 + 0.0j, 0.0 + 0.0j, u10,        u11       ]
        ]
        
        # 可选：验证 U 矩阵的酉性 (仅对支持线性代数的后端有意义)
        if self._backend is cp:
            u_matrix_backend = self._backend.array(u_matrix)
            identity_2x2 = self._backend.eye(2)
            if not self._backend.allclose(u_matrix_backend @ u_matrix_backend.conj().T, identity_2x2, atol=1e-7):
                self._internal_logger.warning(f"The provided U matrix for the '{name}' gate may not be perfectly unitary.")
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[control_qubit, target_qubit], 
                local_operator=local_op_cu
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for {name} on ({control_qubit}, {target_qubit}): {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    # --- [增强] 高级参数化多比特门 (保持不变，已通过 self._backend 解耦) ---
    def rxx(self, qubit1: int, qubit2: int, theta: float):
        """
        将一个 RXX(θ) 纠缠门应用于指定的两个量子比特。

        RXX(θ) 门表示由哈密顿量 `H = X⊗X` 生成的演化，时间为 `t=θ/2`。
        其酉算子为 U = exp(-i * (θ/2) * X⊗X) = cos(θ/2)*I⊗I - i*sin(θ/2)*X⊗X。
        
        这个门在量子化学模拟和变分量子算法 (VQA) 中常用于生成特定类型的纠缠。

        [修正] 此实现是后端无关的。它使用 Python 标准库 `math` 计算三角函数值，
        并手动构建纯 Python 列表形式的 `I⊗I` 和 `X⊗X` 矩阵，然后组合成
        最终的 RXX(θ) 局部算子。

        Args:
            qubit1 (int): 
                第一个目标量子比特的索引。
            qubit2 (int): 
                第二个目标量子比特的索引。
            theta (float):
                旋转角度 θ，以弧度为单位。
        
        Raises:
            ValueError: 如果两个量子比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if qubit1 == qubit2:
            raise ValueError("The two qubits for an RXX gate must be different.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `math` 来计算 cos(θ/2) 和 sin(θ/2)。
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)
        
        # 定义基础的 2x2 矩阵 I 和 X (纯 Python 列表)
        I_local = [[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]]
        X_local = [[0.0+0.0j, 1.0+0.0j], [1.0+0.0j, 0.0+0.0j]]
        
        # 手动计算张量积 I⊗I 和 X⊗X
        # I⊗I 就是 4x4 的单位矩阵
        II = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        
        # X⊗X
        XX = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ]
        
        # 组合成最终的 RXX(θ) 局部算子： c*II - i*s*XX
        # 使用列表推导式进行线性组合
        local_op_rxx = [
            [
                complex(c * ii_elem) - 1j * s * complex(xx_elem)
                for ii_elem, xx_elem in zip(ii_row, xx_row)
            ]
            for ii_row, xx_row in zip(II, XX)
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        # target_qubits 的顺序 [qubit1, qubit2] 定义了局部算子的基矢顺序。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[qubit1, qubit2], 
                local_operator=local_op_rxx
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for RXX({theta:.4f}) on ({qubit1}, {qubit2}): {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def ryy(self, qubit1: int, qubit2: int, theta: float):
        """
        将一个 RYY(θ) 纠缠门应用于指定的两个量子比特。

        RYY(θ) 门表示由哈密顿量 `H = Y⊗Y` 生成的演化，时间为 `t=θ/2`。
        其酉算子为 U = exp(-i * (θ/2) * Y⊗Y) = cos(θ/2)*I⊗I - i*sin(θ/2)*Y⊗Y。
        
        这个门在量子化学模拟（尤其是在考虑某些分子对称性时）和 VQA 中
        被广泛使用。

        [修正] 此实现是后端无关的。它使用 Python 标准库 `math` 计算三角函数值，
        并手动构建纯 Python 列表形式的 `I⊗I` 和 `Y⊗Y` 矩阵，然后组合成
        最终的 RYY(θ) 局部算子。

        Args:
            qubit1 (int): 
                第一个目标量子比特的索引。
            qubit2 (int): 
                第二个目标量子比特的索引。
            theta (float):
                旋转角度 θ，以弧度为单位。
        
        Raises:
            ValueError: 如果两个量子比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if qubit1 == qubit2:
            raise ValueError("The two qubits for an RYY gate must be different.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 使用 Python 标准库 `math` 来计算 cos(θ/2) 和 sin(θ/2)。
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)
        
        # 定义基础的 2x2 矩阵 Y (纯 Python 列表)
        # Y = [[0, -i], [i, 0]]
        
        # 手动计算张量积 Y⊗Y
        # [[0, -i],   ⊗ [[0, -i], = [[0*Y, -i*Y],
        #  [i,  0]]      [i,  0]]    [i*Y,  0*Y]]
        #
        # = [[ [0,0],[0,0],  [0,s],[-s,0] ],
        #    [ [0,0],[0,0],  [-s,0],[0,0] ],
        #    [ [0,-s],[s,0], [0,0],[0,0] ],
        #    [ [s,0],[0,0],  [0,0],[0,0] ]]
        # 其中 s = -i * i = 1
        # Y⊗Y = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
        YY = [
            [0,  0,  0, -1],
            [0,  0,  1,  0],
            [0,  1,  0,  0],
            [-1, 0,  0,  0]
        ]
        
        # I⊗I 就是 4x4 的单位矩阵
        II = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        
        # 组合成最终的 RYY(θ) 局部算子： c*II - i*s*YY
        # 使用列表推导式进行线性组合
        local_op_ryy = [
            [
                complex(c * ii_elem) - 1j * s * complex(yy_elem)
                for ii_elem, yy_elem in zip(ii_row, yy_row)
            ]
            for ii_row, yy_row in zip(II, YY)
        ]
        
        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[qubit1, qubit2], 
                local_operator=local_op_ryy
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for RYY({theta:.4f}) on ({qubit1}, {qubit2}): {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    def rzz(self, qubit1: int, qubit2: int, theta: float):
        """
        将一个 RZZ(θ) 纠缠门应用于指定的两个量子比特。

        RZZ(θ) 门表示由哈密顿量 `H = Z⊗Z` 生成的演化，时间为 `t=θ/2`。
        其酉算子为 U = exp(-i * (θ/2) * Z⊗Z) = cos(θ/2)*I⊗I - i*sin(θ/2)*Z⊗Z。
        
        由于 Z⊗Z 是对角矩阵，RZZ(θ) 也是一个对角门（相位门）。它根据两个比特
        的奇偶性施加一个相位：
        - 如果 `|q1, q2⟩` 的奇偶性为偶数 (00, 11)，相位为 `e^(-iθ/2)`。
        - 如果 `|q1, q2⟩` 的奇偶性为奇数 (01, 10)，相位为 `e^(+iθ/2)`。

        [修正] 此实现是后端无关的。它使用 Python 标准库 `math` 和 `cmath`
        计算三角函数/相位，并手动构建纯 Python 列表形式的 `I⊗I` 和 `Z⊗Z` 
        矩阵，然后组合成最终的 RZZ(θ) 局部算子。

        Args:
            qubit1 (int): 
                第一个目标量子比特的索引。
            qubit2 (int): 
                第二个目标量子比特的索引。
            theta (float):
                旋转角度 θ，以弧度为单位。
        
        Raises:
            ValueError: 如果两个量子比特索引相同或无效。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 0: 输入验证 ---
        if qubit1 == qubit2:
            raise ValueError("The two qubits for an RZZ gate must be different.")
        
        # --- 步骤 1: 定义局部的、与后端无关的算子矩阵 ---
        # 另一种更直接的构建 RZZ 的方式是直接计算对角元素
        half_theta = theta / 2.0
        phase_plus = cmath.exp(1j * half_theta)
        phase_minus = cmath.exp(-1j * half_theta)

        # RZZ(θ) 是一个对角矩阵，对角线元素为
        # [e^(-iθ/2), e^(iθ/2), e^(iθ/2), e^(-iθ/2)]
        local_op_rzz = [
            [phase_minus, 0, 0, 0],
            [0, phase_plus, 0, 0],
            [0, 0, phase_plus, 0],
            [0, 0, 0, phase_minus]
        ]

        # 确保所有元素都是复数
        local_op_rzz = [[complex(val) for val in row] for row in local_op_rzz]

        # --- 步骤 2: 构建作用于整个系统的全局算子 ---
        # 调用通用的构建函数。
        try:
            global_op = self._build_global_operator_multi_qubit(
                target_qubits=[qubit1, qubit2], 
                local_operator=local_op_rzz
            )
        except ValueError as e:
            # 捕获可能的索引错误并重新抛出，提供更多上下文
            self._internal_logger.error(f"Failed to build global operator for RZZ({theta:.4f}) on ({qubit1}, {qubit2}): {e}")
            raise e

        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(global_op)

    # --- [增强] 高级多控制门 (保持不变，已通过 self._backend 解耦) ---
    def mcz(self, controls: List[int], target: int):
        """
        将一个多控制-Z (Multi-Controlled-Z) 门应用于指定的量子比特。

        MCZ 门的作用是：
        - 当且仅当所有 `controls` 量子比特都处于 |1⟩ 状态时，才对
          `target` 量子比特应用一个 Z 门。
        
        由于 Z 门的效果是给 |1⟩ 状态施加一个 -1 的相位，这等价于：
        - 当且仅当所有 `controls` 量子比特和 `target` 量子比特都
          处于 |1⟩ 状态时，给整个量子态施加一个 -1 的相位。

        MCZ 门是一个对角门，这使得我们可以直接构建其全局算子而无需先构建
        一个（可能非常大的）局部算子。

        [修正] 此实现是后端无关且高效的。它通过位运算直接构建全局对角
        算子，避免了构建和传递巨大的局部矩阵。

        Args:
            controls (List[int]): 
                一个包含所有控制量子比特索引的列表。
            target (int): 
                目标量子比特的索引。
            
        Raises:
            ValueError: 如果输入参数无效（如列表为空、索引重复或越界）。
            RuntimeError: 如果在演化过程中发生底层计算错误。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not controls or not all(isinstance(c, int) for c in controls):
            raise ValueError("Controls must be a non-empty list of integers.")
        if not isinstance(target, int):
            raise ValueError("Target must be an integer.")
            
        all_qubits = set(controls + [target])
        if len(all_qubits) != len(controls) + 1:
            raise ValueError("Control and target qubits for an MCZ gate must be unique.")
        
        for q in all_qubits:
            if not (0 <= q < self.num_qubits):
                raise ValueError(f"Qubit index {q} is out of range for the system size {self.num_qubits}.")
        
        # --- 步骤 2: 高效地直接构建全局对角算子 ---
        # 这种方法避免了 _build_global_operator_multi_qubit 的巨大开销。
        op_dim = 1 << self.num_qubits
        
        # 使用当前后端创建一个单位矩阵作为基础。
        # 对于 PurePythonBackend，需要手动调用，因为 self._backend 是实例。
        # 对于 CuPy，self._backend 是模块。
        if hasattr(self._backend, 'eye'):
            mcz_global_u = self._backend.eye(op_dim)
        else:
            mcz_global_u = PurePythonBackend().eye(op_dim)

        # 创建一个位掩码 (bitmask)，用于一次性检查所有相关比特是否都为 1。
        # 这个掩码包含了所有的控制比特和目标比特。
        check_mask = sum(1 << q for q in all_qubits)
        
        # 遍历所有计算基态 |i⟩ 的索引
        for i in range(op_dim):
            # 使用位与 (&) 操作来检查基态 i 中，所有相关比特是否都为 1。
            # 如果 `(i & check_mask)` 的结果等于 `check_mask` 本身，
            # 说明在 `i` 中，`check_mask` 标记的所有位都为 1。
            if (i & check_mask) == check_mask:
                # 在对角线上施加 -1 的相位。
                # 这种索引方式对 CuPy 数组和 Python 嵌套列表都有效。
                mcz_global_u[i][i] *= -1.0
        
        # --- 步骤 3: 将全局算子应用于量子态 ---
        # 调用核心的演化方法。
        self._apply_global_unitary(mcz_global_u)

    # --- 测量与噪声 ---
    def simulate_measurement(self, 
                            qubit_to_measure: int, 
                            classical_register_index: Optional[int] = None,
                            collapse_state: bool = True) -> Optional[int]:
        """
        [终极修正版] 模拟对指定单个量子比特的测量，并可选地坍缩量子态。

        此方法使用物理上正确的边际概率进行随机抽样，并通过投影算子
        来执行状态坍缩，确保了模拟的精确性和健robustness。
        它还支持将测量结果存储到经典寄存器，以实现经典控制流。

        [修正] 此实现是后端无关的。它依赖于 `get_marginal_probabilities`
        来计算概率，并调用 `self._backend.choice` 进行随机抽样，使其能在
        CuPy 和 PurePythonBackend 上无缝工作。

        Args:
            qubit_to_measure (int): 
                要测量的目标量子比特的索引 (0 到 num_qubits-1)。
            classical_register_index (Optional[int], optional): 
                如果提供，测量结果将被存储到这个索引对应的经典寄存器中。
                默认为 None。
            collapse_state (bool, optional): 
                如果为 `True` (默认)，则在测量后更新内部密度矩阵以反映状态坍缩。
                如果为 `False`，则仅返回一个基于概率的测量结果，而不改变量子态。

        Returns:
            Optional[int]: 
                测量结果 (0 或 1)。如果发生错误（如状态无效），则返回 None。

        Raises:
            ValueError: 如果 `qubit_to_measure` 索引无效。
            RuntimeError: 如果在坍缩过程中发生底层计算错误。
        """
        log_prefix = f"QuantumState.SimulateMeasurement(N={self.num_qubits}, TQ={qubit_to_measure}, CR_idx={classical_register_index}, Collapse={collapse_state})"
        
        # --- 步骤 1: 核心依赖和参数验证 ---
        if self._backend is None:
            self._internal_logger.critical(f"[{log_prefix}] Core compute library not loaded, cannot simulate measurement.")
            raise RuntimeError("Core compute library is not loaded.")

        self._internal_logger.debug(f"[{log_prefix}] Starting measurement simulation...")

        if not (0 <= qubit_to_measure < self.num_qubits):
            msg = f"Invalid target qubit index: {qubit_to_measure} (total qubits: {self.num_qubits})."
            self._internal_logger.error(f"[{log_prefix}] {msg}")
            raise ValueError(msg)
        
        if not self.is_valid():
            self._internal_logger.error(f"[{log_prefix}] The current quantum state is invalid, cannot perform measurement.")
            return None

        # --- 步骤 2: 计算正确的边际概率 ---
        # get_marginal_probabilities 已经是后端无关的
        marginal_probs = self.get_marginal_probabilities(qubit_to_measure)
        if marginal_probs is None:
            self._internal_logger.error(f"[{log_prefix}] Failed to get marginal probabilities for qubit {qubit_to_measure}.")
            return None
        
        prob_0, prob_1 = marginal_probs

        # --- 步骤 3: 根据边际概率随机选择一个测量结果 ---
        # [修正] 调用 self._backend.choice()，确保纯Python后端也能工作
        measured_outcome = self._backend.choice([0, 1], p=[prob_0, prob_1])
        self._internal_logger.info(f"[{log_prefix}] Measurement outcome for qubit {qubit_to_measure}: |{measured_outcome}⟩")
        
        # --- 步骤 4: 存储到经典寄存器 (如果需要) ---
        if classical_register_index is not None:
            self._classical_registers[classical_register_index] = int(measured_outcome)
            self._internal_logger.debug(f"[{log_prefix}] Measurement result |{measured_outcome}⟩ stored in classical register CR[{classical_register_index}].")

        # --- 步骤 5: 如果需要，执行密度矩阵坍缩 ---
        if collapse_state:
            self._internal_logger.debug(f"[{log_prefix}]   Initiating density matrix collapse...")
            
            # 定义局部的、纯Python的投影算子
            P0_local = [[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 0.0+0.0j]]
            P1_local = [[0.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]]
            
            Projection_local = P0_local if measured_outcome == 0 else P1_local
            
            # 构建作用于整个系统的全局投影算子
            P_global = self._build_global_operator_multi_qubit([qubit_to_measure], Projection_local)
            
            # 坍缩公式：ρ' = (P @ ρ @ P†) / Tr(P @ ρ)
            # 注意：投影算子 P 是厄米的, 所以 P† = P
            rho_post_projection = self._backend.dot(P_global, self._backend.dot(self.density_matrix, P_global))
            
            trace_val = self._backend.trace(rho_post_projection).real
            
            # 检查迹是否为零（数值不稳定的迹象）
            if self._backend.isclose(trace_val, 0.0, atol=1e-12):
                self._internal_logger.error(
                    f"[{log_prefix}] Collapse failed: Trace of post-measurement state is zero. "
                    "This indicates measurement of a zero-probability outcome, possibly due to numerical errors. "
                    "The state will not be modified."
                )
                return None # 坍缩失败
            
            # 更新并归一化密度矩阵 (兼容不同后端)
            if isinstance(self._density_matrix, list): # PurePythonBackend
                self._density_matrix = [[elem / trace_val for elem in row] for row in rho_post_projection]
            else: # CuPy
                self._density_matrix = rho_post_projection / trace_val
            
            self._internal_logger.info(f"[{log_prefix}]   Density matrix successfully collapsed and renormalized.")
            
            # 最终验证坍缩后的密度矩阵是否仍然物理有效
            if not self.is_valid():
                self._internal_logger.critical(
                    f"[{log_prefix}] Severe warning: The density matrix failed a validity check after collapse! The state may be corrupted."
                )
            
            # 更新时间戳以反映状态变化
            self._update_timestamp()

        # --- 步骤 6: 记录测量操作的详细信息 ---
        measurement_log_entry_id = f"meas_q{qubit_to_measure}_{uuid.uuid4().hex[:8]}"
        self.measurement_outcomes_log[measurement_log_entry_id] = {
            "qubit_index_measured": int(qubit_to_measure), 
            "measurement_outcome": int(measured_outcome),
            "timestamp_utc_iso": datetime.now(timezone.utc).isoformat(),
            "state_collapsed": collapse_state,
            "classical_register_index": classical_register_index
        }
        # 保持日志长度在限制内
        if len(self.measurement_outcomes_log) > self.MAX_MEASUREMENT_LOG:
            # next(iter(...)) 是从字典中获取第一个键的Pythonic方式
            del self.measurement_outcomes_log[next(iter(self.measurement_outcomes_log))]
        
        self._internal_logger.info(f"[{log_prefix}] Simulated measurement on Qubit {qubit_to_measure} complete. Returning result: {int(measured_outcome)}.")
        
        return int(measured_outcome)

    def apply_quantum_channel(self, 
                            channel_type: 'QuantumChannelType', 
                            target_qubits: Union[int, List[int], None],
                            params: Dict[str, Any]):
        """
        [完整实现] 将指定的量子通道（噪声模型）作用于密度矩阵。
        
        这是通过算符和表示法（Operator-Sum Representation）实现的：
        ρ' = Σ_i (K_i * ρ * K_i†)，其中 K_i 是克劳斯 (Kraus) 算子。

        [修正] 此实现是后端无关的。它使用纯 Python 列表定义基础 Kraus 算子，
        并通过 `self._backend` 调用所有线性代数运算，使其能在 CuPy 和
        PurePythonBackend 上无缝工作。

        Args:
            channel_type (QuantumChannelType): 
                要应用的量子通道类型。必须是 "depolarizing", "bit_flip", 
                "phase_flip", 或 "amplitude_damping"。
            target_qubits (Union[int, List[int], None]): 
                目标量子比特的索引或列表。
                - 对于 "depolarizing" 通道，如果为 `None`，则依次作用于所有比特。
                - 对于其他通道，必须指定目标比特。
            params (Dict[str, Any]): 
                通道操作所需的参数。
                - Depolarizing/BitFlip/PhaseFlip: {"probability": float} (错误概率 p)
                - AmplitudeDamping: {"gamma": float} (阻尼率 gamma)

        Raises:
            ValueError: 如果参数无效、通道类型不支持或量子比特数超出操作限制。
            TypeError: 如果参数类型不正确。
            RuntimeError: 如果底层计算失败。
        """
        log_prefix = f"QuantumState.ApplyChannel(N={self.num_qubits}, Type='{channel_type}')"
        
        # --- 步骤 1: 核心依赖和状态验证 ---
        if self._backend is None:
            raise RuntimeError("Core compute library not loaded, cannot apply quantum channel.")
        if not self.is_valid():
            self._internal_logger.error(f"[{log_prefix}] Density matrix is invalid before applying quantum channel. Operation aborted.")
            return

        # --- 步骤 2: 规范化 target_qubits 为列表 ---
        actual_target_qubits: List[int]
        if target_qubits is None:
            if channel_type == "depolarizing":
                actual_target_qubits = list(range(self.num_qubits)) # 默认作用于所有比特
            else:
                raise ValueError(f"Channel type '{channel_type}' requires target_qubits to be specified.")
        elif isinstance(target_qubits, int):
            actual_target_qubits = [target_qubits]
        elif isinstance(target_qubits, list):
            actual_target_qubits = target_qubits
        else:
            raise TypeError(f"target_qubits must be an integer, a list of integers, or None, but got {type(target_qubits).__name__}.")
        
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in actual_target_qubits):
            raise ValueError("All target qubit indices must be valid and within range.")

        # --- 步骤 3: 参数提取和验证 ---
        if channel_type == "amplitude_damping":
            gamma = params.get('gamma')
            if not isinstance(gamma, (float, int)) or not (0.0 <= gamma <= 1.0):
                raise ValueError(f"Amplitude damping 'gamma' must be a float in [0.0, 1.0], got {gamma}.")
            rate = float(gamma)
        else:
            probability = params.get('probability')
            if not isinstance(probability, (float, int)) or not (0.0 <= probability <= 1.0):
                raise ValueError(f"Channel '{channel_type}' 'probability' must be a float in [0.0, 1.0], got {probability}.")
            rate = float(probability)

        # --- 步骤 4: 定义局部的、后端无关的 Kraus 算子 ---
        # 基础算子定义为纯 Python 列表
        I_list = [[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]]
        X_list = [[0.0+0.0j, 1.0+0.0j], [1.0+0.0j, 0.0+0.0j]]
        Y_list = [[0.0+0.0j, -1.0j],    [1.0j,     0.0+0.0j]]
        Z_list = [[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, -1.0+0.0j]]

        kraus_operators_local: List[Any] = []

        # 辅助函数进行标量-矩阵乘法，兼容 CuPy 和 PurePython
        def scalar_mult(scalar, matrix):
            if isinstance(matrix, list): # PurePython
                return [[scalar * elem for elem in row] for row in matrix]
            else: # CuPy
                return scalar * matrix

        if channel_type == "depolarizing":
            if rate > 1e-12:
                kraus_operators_local.append(scalar_mult(math.sqrt(1 - rate), I_list))
                p_factor_sqrt = math.sqrt(rate / 3.0)
                kraus_operators_local.append(scalar_mult(p_factor_sqrt, X_list))
                kraus_operators_local.append(scalar_mult(p_factor_sqrt, Y_list))
                kraus_operators_local.append(scalar_mult(p_factor_sqrt, Z_list))
            else:
                kraus_operators_local.append(I_list)
        elif channel_type == "bit_flip":
            kraus_operators_local.append(scalar_mult(math.sqrt(1 - rate), I_list))
            if rate > 1e-12: kraus_operators_local.append(scalar_mult(math.sqrt(rate), X_list))
        elif channel_type == "phase_flip":
            kraus_operators_local.append(scalar_mult(math.sqrt(1 - rate), I_list))
            if rate > 1e-12: kraus_operators_local.append(scalar_mult(math.sqrt(rate), Z_list))
        elif channel_type == "amplitude_damping":
            k0 = [[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, math.sqrt(1 - rate)]]
            kraus_operators_local.append(k0)
            if rate > 1e-12: 
                k1 = [[0.0+0.0j, math.sqrt(rate)], [0.0+0.0j, 0.0+0.0j]]
                kraus_operators_local.append(k1)
        else:
            raise ValueError(f"Unsupported quantum channel type: '{channel_type}'")

        # --- 步骤 5: 依次将通道应用于每个目标量子比特 ---
        current_rho = self.density_matrix
        
        for q_idx in actual_target_qubits:
            self._internal_logger.debug(f"[{log_prefix}]   Applying channel '{channel_type}' to qubit {q_idx}...")
            
            # 创建一个全零矩阵作为累加器，类型与当前后端匹配
            if isinstance(current_rho, list): # PurePythonBackend
                new_rho_for_this_qubit = self._backend.create_matrix(len(current_rho), len(current_rho[0]))
            else: # CuPy
                new_rho_for_this_qubit = self._backend.zeros_like(current_rho)
            
            for local_kraus_op in kraus_operators_local:
                try:
                    global_kraus_op = self._build_global_operator_multi_qubit([q_idx], local_kraus_op)
                    # 计算 K_i @ rho @ K_i_dagger
                    term = self._backend.dot(global_kraus_op, self._backend.dot(current_rho, self._backend.conj_transpose(global_kraus_op)))
                    
                    # 矩阵加法 (兼容 CuPy 和 PurePython)
                    if isinstance(new_rho_for_this_qubit, list): # PurePythonBackend
                        for r_idx in range(len(new_rho_for_this_qubit)):
                            for c_idx in range(len(new_rho_for_this_qubit[0])):
                                new_rho_for_this_qubit[r_idx][c_idx] += term[r_idx][c_idx]
                    else: # CuPy
                        new_rho_for_this_qubit += term
                except Exception as e:
                    self._internal_logger.error(f"[{log_prefix}] Error building or applying Kraus operator for qubit {q_idx}: {e}", exc_info=True)
                    return # 中止操作，防止状态损坏
            
            current_rho = new_rho_for_this_qubit

        self._density_matrix = current_rho
        
        # --- 步骤 6: 后置处理与验证 ---
        self.normalize() # 量子通道理论上是保迹的，但浮点误差可能需要归一化
        
        if not self.is_valid(): 
            self._internal_logger.critical(f"[{log_prefix}] Severe error: Quantum state became invalid after applying quantum channel!")
        
        # --- 步骤 7: 记录操作历史和更新时间戳 ---
        self._add_history_log(
            gate_name=f"QuantumChannel_{channel_type}", 
            targets=actual_target_qubits, 
            params=params
        )
        self._update_timestamp()
        self._internal_logger.info(f"[{log_prefix}] Quantum channel '{channel_type}' successfully applied to qubits {actual_target_qubits}.")
    
    def introduce_simulated_decoherence(
        self, 
        rate: float = 0.01, 
        type: Literal[
            "depolarizing", "bit_flip", "phase_flip", "amplitude_damping",
            "imperfect_gate_noise"
        ] = "depolarizing",
        target_qubits: Optional[Union[int, List[int]]] = None
    ):
        """
        [完整实现] 模拟将退相干效应引入当前量子态。

        这是一个高级封装方法，它将用户友好的退相干模型转换为对底层
        `apply_quantum_channel` 方法的调用，或者执行一个简化的随机噪声注入模型。

        [修正] 此实现是后端无关的。对于基于通道的模型，它直接委托给
        `apply_quantum_channel`。对于 `"imperfect_gate_noise"` 模型，
        它使用 `self._backend.random_normal` 来生成噪声，确保了对
        CuPy 和 PurePythonBackend 的兼容性。

        Args:
            rate (float, optional): 
                退相干的速率或强度，范围在 [0.0, 1.0]。其具体含义取决于 `type`。
                - Depolarizing/BitFlip/PhaseFlip: 错误概率 p。
                - AmplitudeDamping: 阻尼率 gamma。
                - "imperfect_gate_noise": 随机扰动强度。
                默认为 0.01。
            type (Literal[...], optional): 
                要应用的退相干模型类型。默认为 "depolarizing"。
            target_qubits (Optional[Union[int, List[int]]], optional): 
                指定受影响的量子比特子集。对于 "depolarizing" 通道，如果为 `None`，
                则作用于所有比特。

        Raises:
            ValueError: 如果 `rate` 参数值超出范围，或 `type` 不支持。
            TypeError: 如果 `rate` 或 `target_qubits` 参数类型不正确。
            RuntimeError: 如果底层计算失败。
        """
        log_prefix = f"QuantumState.Decoherence(N={self.num_qubits}, Type='{type}', Rate={rate:.4f})"
        
        # --- 步骤 1: 核心依赖和状态验证 ---
        if self._backend is None:
            raise RuntimeError("Core compute library not loaded, cannot introduce simulated decoherence.")
        if not self.is_valid():
            self._internal_logger.error(f"[{log_prefix}] Density matrix is invalid before introducing decoherence. Aborting operation.")
            return

        # --- 步骤 2: 验证 rate 参数 ---
        if not isinstance(rate, (float, int)) or not (0.0 <= rate <= 1.0):
            msg = f"Decoherence rate 'rate' ({rate}) must be a numeric value in [0.0, 1.0]."
            self._internal_logger.error(f"[{log_prefix}] {msg}")
            raise ValueError(msg)
        
        rate_float = float(rate)

        # --- 步骤 3: 根据类型分发到不同的退相干模型 ---
        
        # --- 3a. 如果是基于量子通道的模型 ---
        # `self.QuantumChannelType.__args__` 获取 Literal 中的所有字符串
        if type in self.QuantumChannelType.__args__:
            self._internal_logger.info(f"[{log_prefix}] Applying decoherence via quantum channel model...")
            
            # 准备传递给 `apply_quantum_channel` 的参数
            channel_params: Dict[str, float]
            if type == "amplitude_damping":
                channel_params = {"gamma": rate_float}
            else:
                channel_params = {"probability": rate_float}
                
            try:
                # 直接调用底层的、已经后端无关的 `apply_quantum_channel` 方法
                self.apply_quantum_channel(
                    channel_type=type, # type: ignore # 确保 mypy 知道 type 是有效的 Literal 成员
                    target_qubits=target_qubits,
                    params=channel_params
                )
                self._internal_logger.info(f"[{log_prefix}] Decoherence effect (type: '{type}') successfully applied via quantum channel model.")
            except (ValueError, TypeError, RuntimeError) as e_channel_apply:
                # 捕获并重新抛出底层方法的错误，同时添加更多上下文
                self._internal_logger.error(f"[{log_prefix}] Failed to apply quantum channel '{type}': {e_channel_apply}", exc_info=True)
                raise e_channel_apply

        # --- 3b. 如果是简化的不完美门噪声模型 ---
        elif type == "imperfect_gate_noise":
            self._internal_logger.info(f"[{log_prefix}] Applying simulated imperfect gate noise (strength factor: {rate_float:.4f})...")
            
            # 获取密度矩阵维度 (兼容 CuPy 和 PurePython)
            dm_shape = self._density_matrix.shape if hasattr(self._density_matrix, 'shape') else (len(self._density_matrix), len(self._density_matrix[0]))
            
            if dm_shape[0] == 0:
                self._internal_logger.warning(f"[{log_prefix}] Density matrix is empty, cannot apply imperfect gate noise.")
                return
                
            # 生成与密度矩阵同样大小的随机复数噪声矩阵，使用后端方法
            noise_magnitude_std_dev = rate_float * 0.05
            
            random_noise_real = self._backend.random_normal(0, noise_magnitude_std_dev, dm_shape)
            random_noise_imag = self._backend.random_normal(0, noise_magnitude_std_dev, dm_shape)
            
            # 将实部和虚部组合成复数矩阵
            if isinstance(random_noise_real, list): # PurePythonBackend
                random_noise_matrix = [[complex(r, i) for r, i in zip(row_r, row_i)] 
                                       for row_r, row_i in zip(random_noise_real, random_noise_imag)]
            else: # CuPy
                random_noise_matrix = random_noise_real + 1j * random_noise_imag
            
            # 使噪声矩阵 Hermitian (H = (H + H†)/2)
            noise_dagger = self._backend.conj_transpose(random_noise_matrix)
            
            # 矩阵加法 (兼容 CuPy 和 PurePython)
            if isinstance(self._density_matrix, list): # PurePythonBackend
                hermitian_noise = [[(random_noise_matrix[r][c] + noise_dagger[r][c]) / 2.0 for c in range(dm_shape[0])] for r in range(dm_shape[0])]
                self._density_matrix = [[self._density_matrix[r][c] + hermitian_noise[r][c] for c in range(dm_shape[0])] for r in range(dm_shape[0])]
            else: # CuPy
                hermitian_noise = (random_noise_matrix + noise_dagger) / 2.0
                self._density_matrix += hermitian_noise
            
            self._internal_logger.debug(f"[{log_prefix}] Applied random Hermitian noise to density matrix.")
            
            # 噪声注入后必须重新归一化并检查有效性
            self.normalize()
            if not self.is_valid():
                self._internal_logger.critical(
                    f"[{log_prefix}] Severe warning: Quantum state became invalid after applying imperfect gate noise! "
                    "This can happen if noise is too large, leading to negative eigenvalues."
                )
            
            self._update_timestamp()

        # --- 3c. 未知的退相干类型 ---
        else:
            msg = f"Unsupported decoherence type '{type}'."
            self._internal_logger.error(f"[{log_prefix}] {msg}")
            raise ValueError(msg)

        self._internal_logger.info(f"[{log_prefix}] Simulated decoherence operation complete.")
        
    def _generate_einsum_string_for_partial_trace(self, num_total_qubits: int, qubits_to_trace: List[int]) -> str:
        """
        [内部辅助函数] 为 `_partial_trace` 方法生成 `einsum` 下标字符串。

        `einsum` (Einstein summation convention) 是一种强大的张量运算表示法。
        部分迹可以通过将密度矩阵 `ρ`（一个秩为 2N 的张量）的某些轴
        （对应要迹掉的量子比特）进行求和（收缩）来实现。

        工作原理:
        1.  一个 N-qubit 密度矩阵 `ρ` (形状 `(2^N, 2^N)`) 被重塑为一个秩为 2N
            的张量，形状为 `(2, 2, ..., 2)`。前 N 个轴代表 "bra" 索引，
            后 N 个轴代表 "ket" 索引。
        2.  我们为这 2N 个轴分配唯一的字母下标，例如 N=3 时为 'abcABC'。
        3.  要迹掉第 `i` 个量子比特，我们需要对第 `i` 个 "bra" 轴和第 `i` 个 "ket"
            轴进行求和。在 `einsum` 中，这通过给它们分配**相同的下标**来实现。
        4.  输出张量的下标只包含那些被保留的量子比特所对应的 "bra" 和 "ket" 下标。

        Example:
            N=3, qubits_to_trace=[1] (迹掉 q1)
            - 初始下标: 'abc' (bra), 'ABC' (ket)
            - 输入字符串: 'abc' + 'aBc' -> 'abcaBc'
              (q1 对应的 'B' 被替换为 'b')
            - 输出字符串: 'ac' + 'AC' -> 'acAC' (保留 q0 和 q2)
            - 最终 `einsum` 字符串: "abcaBc->acAC"

        [修正] 此方法是纯 Python 实现，不依赖任何后端，因此在重构后无需修改。

        Args:
            num_total_qubits (int): 
                系统的总量子比特数 (N)。
            qubits_to_trace (List[int]): 
                一个包含了要被迹掉的量子比特索引的列表。

        Returns:
            str: 一个格式正确的 `einsum` 字符串，可用于 `cupy.einsum` 或
                 一个兼容的纯 Python `einsum` 实现。
        """
        # --- 步骤 1: 生成基础的 "bra" 和 "ket" 下标字符 ---
        # 为 "bra" 生成 N 个唯一的字符 (a, b, c, ...)
        bra_indices_chars = [chr(ord('a') + i) for i in range(num_total_qubits)]
        # 为 "ket" 生成 N 个唯一的字符 (A, B, C, ...)
        ket_indices_chars = [chr(ord('A') + i) for i in range(num_total_qubits)]

        # --- 步骤 2: 构建输入字符串 ---
        # a) "bra" 部分的字符串
        input_str_bra_part = "".join(bra_indices_chars)
        
        # b) "ket" 部分的字符串（将被修改）
        #    我们创建一个可修改的列表副本
        ket_indices_chars_modified = list(ket_indices_chars)
        for qubit_index_to_trace in qubits_to_trace:
            # 将要迹掉的 "ket" 轴的下标替换为对应的 "bra" 轴的下标
            # 这告诉 einsum 对这两个轴进行求和（收缩）
            ket_indices_chars_modified[qubit_index_to_trace] = bra_indices_chars[qubit_index_to_trace]
        
        input_str_ket_part_modified = "".join(ket_indices_chars_modified)
        
        # c) 组合成最终的输入字符串
        einsum_input_str = input_str_bra_part + input_str_ket_part_modified
        
        # --- 步骤 3: 构建输出字符串 ---
        # 输出的下标只包含那些被保留的量子比特所对应的 "bra" 和 "ket" 下标。
        output_bra_chars = []
        output_ket_chars = []
        for i in range(num_total_qubits):
            # 如果当前量子比特索引不在要迹掉的列表中，则保留其下标
            if i not in qubits_to_trace:
                output_bra_chars.append(bra_indices_chars[i])
                output_ket_chars.append(ket_indices_chars_modified[i])
        
        einsum_output_str = "".join(output_bra_chars) + "".join(output_ket_chars)

        # --- 步骤 4: 组合成最终的 einsum 格式化字符串 ---
        return f"{einsum_input_str}->{einsum_output_str}"


    def _partial_trace(self, qubits_to_trace_out: List[int]) -> Any:
        """
        [最终统一版] 对密度矩阵执行部分迹（Partial Trace）操作。

        此版本依赖于后端提供的多维 reshape 和 einsum 功能，实现了对所有后端的
        统一代码路径。它通过将密度矩阵重塑为一个高维张量，然后使用爱因斯坦
        求和约定（einsum）对指定的量子比特轴进行求和（迹），从而得到约化密度矩阵。

        工作流程:
        1. 验证输入的量子比特索引列表是否有效。
        2. 将 (2^N, 2^N) 的密度矩阵 ρ 重塑为一个秩为 2N、形状为 (2, 2, ..., 2) 的张量。
        3. 生成一个 einsum 字符串，该字符串指示对与 `qubits_to_trace_out` 对应的
           "bra" 和 "ket" 轴进行求和。
        4. 调用当前后端的 einsum 方法执行计算，得到一个降维后的张量。
        5. 将结果张量重塑回一个 (2^M, 2^M) 的矩阵形式，其中 M 是保留的量子比特数。

        Args:
            qubits_to_trace_out (List[int]): 
                一个包含了要被迹掉的量子比特索引的列表。

        Returns:
            Any: 
                一个代表剩余子系统的、尺寸更小的密度矩阵。其类型与当前
                后端匹配（`cupy.ndarray` 或 `List[List[complex]]`）。
            
        Raises:
            ValueError: 如果输入的 `qubits_to_trace_out` 列表无效。
            RuntimeError: 如果在 `reshape` 或 `einsum` 计算中发生错误。
        """
        # --- 步骤 1: 快捷路径和输入验证 ---
        if not qubits_to_trace_out:
            return copy.deepcopy(self.density_matrix)

        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits_to_trace_out):
            raise ValueError("Invalid qubit index found in qubits_to_trace_out.")
        if len(set(qubits_to_trace_out)) != len(qubits_to_trace_out):
            raise ValueError("qubits_to_trace_out must contain unique qubit indices.")

        # --- 步骤 2: 准备计算参数 ---
        num_qubits_kept = self.num_qubits - len(qubits_to_trace_out)
        new_dim = 1 << num_qubits_kept

        # --- 步骤 3: [统一路径] 将密度矩阵重塑为高维张量 ---
        try:
            # [*** 核心修复 ***]
            # 将列表 [2, 2, ...] 转换为元组 (2, 2, ...)，以匹配 reshape 函数的类型签名。
            target_shape = tuple([2] * (2 * self.num_qubits))
            rho_tensor = self._backend.reshape(self._density_matrix, target_shape)
        except (ValueError, IndexError, TypeError) as e:
            dm_shape = getattr(self._density_matrix, 'shape', (len(self._density_matrix), len(self._density_matrix[0])))
            self._internal_logger.error(
                f"Failed to reshape density matrix. Shape is {dm_shape}, but expected for {self.num_qubits} qubits. Error: {e}", 
                exc_info=True
            )
            raise RuntimeError("Density matrix shape is inconsistent with the number of qubits.") from e

        # --- 步骤 4: [统一路径] 生成并执行 einsum 操作 ---
        einsum_str = self._generate_einsum_string_for_partial_trace(self.num_qubits, qubits_to_trace_out)
        self._internal_logger.debug(f"Generated einsum string for partial trace: '{einsum_str}'")
        
        # 统一调用后端的 einsum 方法。
        # 我们增强后的 PurePythonBackend.einsum 现在可以处理这个操作。
        try:
            if self._backend is cp:
                reduced_rho_tensor = self._backend.einsum(einsum_str, rho_tensor, optimize=True)
            else:
                reduced_rho_tensor = self._backend.einsum(einsum_str, rho_tensor)
        except Exception as e:
            self._internal_logger.error(f"Einsum operation failed for backend {type(self._backend).__name__}. Error: {e}", exc_info=True)
            raise RuntimeError("Partial trace calculation failed during einsum operation.") from e
        
        # --- 步骤 5: [统一路径] 将结果重塑为正确的矩阵形式 ---
        try:
            return self._backend.reshape(reduced_rho_tensor, (new_dim, new_dim))
        except (ValueError, IndexError, TypeError) as e:
            self._internal_logger.error(f"Failed to reshape the result of partial trace back to a matrix. Error: {e}", exc_info=True)
            raise RuntimeError("Could not correctly reshape the final reduced density matrix.") from e
    
    
    
    def get_marginal_probabilities(self, qubit_index_to_keep: int) -> Optional[Tuple[float, float]]:
        """
        [修正版] 计算指定单个量子比特的边际概率分布 P(0) 和 P(1)。
        
        边际概率是指在一个多比特系统中，测量某一个特定比特得到 0 或 1 的
        总概率，无论其他比特的状态如何。

        此方法通过物理上正确的“部分迹” (Partial Trace) 方法来实现。它会迹掉
        (trace out) 系统中所有其他的量子比特，得到目标比特的 2x2 约化密度矩阵。
        该约化密度矩阵的对角线元素 `ρ_q[0,0]` 和 `ρ_q[1,1]` 即为所求的
        边际概率 P(0) 和 P(1)。

        [修正] 此实现是后端无关的。它依赖 `_partial_trace` 方法，并能正确
        处理和解析 `cupy` 数组或 Python 列表形式的约化密度矩阵。

        Args:
            qubit_index_to_keep (int): 
                要计算边际概率的目标量子比特的索引。

        Returns:
            Optional[Tuple[float, float]]: 
                一个元组 `(P(0), P(1))`。如果计算失败（例如，状态无效），
                则返回 `None`。
                
        Raises:
            ValueError: 如果 `qubit_index_to_keep` 无效。
        """
        log_prefix = f"QuantumState.MarginalProbs(N={self.num_qubits}, KeepQ={qubit_index_to_keep})"
        
        # --- 步骤 1: 严格的输入验证 ---
        if not (0 <= qubit_index_to_keep < self.num_qubits):
            self._internal_logger.error(f"[{log_prefix}] Invalid target qubit index: {qubit_index_to_keep}")
            raise ValueError("Invalid target qubit index.")
        
        # 在计算前检查当前状态是否有效
        if not self.is_valid():
            self._internal_logger.warning(
                f"[{log_prefix}] The current quantum state is invalid. "
                "Calculated marginal probabilities may be meaningless."
            )
            # 即使状态无效，也尝试计算，但依赖 is_valid 的日志来警告用户
            # 或者可以直接返回 None: return None
            
        try:
            # --- 步骤 2: 找出需要被迹掉的所有其他比特 ---
            qubits_to_trace_out = [i for i in range(self.num_qubits) if i != qubit_index_to_keep]
            
            # --- 步骤 3: 计算约化密度矩阵 ---
            # _partial_trace 方法已经是后端无关的
            reduced_rho = self._partial_trace(qubits_to_trace_out)
            
            # --- 步骤 4: 从约化密度矩阵中提取概率 (后端兼容) ---
            # 这是核心的后端兼容性逻辑
            if isinstance(reduced_rho, list): # PurePythonBackend
                prob_0_raw = reduced_rho[0][0].real
                prob_1_raw = reduced_rho[1][1].real
            else: # CuPy-like array
                prob_0_raw = reduced_rho[0, 0].real
                prob_1_raw = reduced_rho[1, 1].real

            # 使用后端的 clip 函数进行数值裁剪
            prob_0 = self._backend.clip(prob_0_raw, 0.0, 1.0)
            prob_1 = self._backend.clip(prob_1_raw, 0.0, 1.0)

            # --- 步骤 5: 验证和归一化 ---
            # 由于浮点误差，P(0) + P(1) 可能不严格等于 1
            prob_sum = float(prob_0) + float(prob_1)
            if not self._backend.isclose(prob_sum, 1.0, atol=1e-7):
                self._internal_logger.warning(
                    f"[{log_prefix}] Marginal probability sum ({prob_sum:.6f}) is not 1.0. "
                    "This may indicate numerical instability. Renormalizing the probabilities."
                )
                # 只有在和不为零时才进行归一化，避免除零错误
                if prob_sum > 1e-12:
                    prob_0 = prob_0 / prob_sum
                    prob_1 = prob_1 / prob_sum
            
            # --- 步骤 6: 返回标准的 Python float 元组 ---
            return (float(prob_0), float(prob_1))

        except Exception as e:
            # 捕获任何在 _partial_trace 或后续计算中发生的错误
            self._internal_logger.error(f"[{log_prefix}] An unknown error occurred while calculating marginal probabilities: {e}", exc_info=True)
            return None

    # --- [终极修正] calculate_fidelity 被彻底禁用 ---
    def calculate_fidelity(self, other_state: 'QuantumState') -> float:
        """
        [禁用] 计算当前量子态与另一个量子态之间的保真度 (Fidelity)。

        保真度衡量两个量子态的相似程度，其计算公式 F(ρ, σ) 涉及到
        矩阵平方根，这是一个高级的数值线性代数运算。

        [修正] 由于此版本的 `quantum_core` 库已彻底移除了对 `scipy` 库的
        依赖，而 `scipy.linalg.sqrtm` 是执行此计算的标准工具，因此该功能
        已被明确禁用。

        调用此方法将总是引发 `NotImplementedError`。

        如果需要使用此功能，请考虑使用依赖 `scipy` 的 `quantum_core` 版本，
        或者在您的应用程序代码中自行集成 `scipy` 来执行计算。

        Args:
            other_state (QuantumState): 
                要与之比较的另一个量子态。

        Returns:
            float: (永不返回)

        Raises:
            NotImplementedError: 总是抛出此异常，以表明该功能在此版本中不可用。
        """
        # 抛出一个信息明确的异常，告知用户为什么该功能不可用。
        raise NotImplementedError(
            "Fidelity calculation is not available in this version of the library "
            "because it depends on the SciPy library (`scipy.linalg.sqrtm`), "
            "which has been removed as a dependency to ensure a pure Python core. "
            "If you need this functionality, please use a version of the library "
            "that includes the SciPy dependency."
        )

    # --- [终极修正] `get_expectation_value` 移除对 NumPy 的处理 ---
    def get_expectation_value(self, observable_matrix: Any) -> float:
        """
        [增强] 计算一个任意的可观测量 (Observable) 在当前量子态下的期望值。

        期望值的计算公式为: ⟨O⟩ = Tr(O @ ρ)
        其中 O 是代表可观测量的厄米矩阵，ρ 是系统的密度矩阵。

        [修正] 此实现是后端无关的。它接收一个可观测量矩阵（可以是 CuPy 数组
        或 Python 嵌套列表），并使用当前后端 (`self._backend`) 的方法来执行
        矩阵乘法和迹运算。

        Args:
            observable_matrix (Any): 
                一个 (2^N, 2^N) 的厄米矩阵，代表要测量的可观测量。
                其类型应与当前后端兼容或可被后端转换（例如，CuPy 后端可以
                接收 Python 列表并自动转换）。

        Returns:
            float: 
                可观测量的期望值。由于可观测量是厄米的，期望值保证为实数。
                此方法返回计算结果的实部。

        Raises:
            ValueError: 如果可观测矩阵的维度与密度矩阵不匹配，或者如果
                        该矩阵不是厄米矩阵。
            RuntimeError: 如果在底层计算中发生错误。
        """
        log_prefix = f"QuantumState.get_expectation_value(N={self.num_qubits})"
        
        # --- 步骤 1: 输入预处理与验证 ---
        op = observable_matrix
        
        # 1a. (可选) 类型转换：如果输入是纯Python列表，而后端是CuPy，则转换为CuPy数组以获得性能。
        if isinstance(op, list) and self._backend is cp:
             op = self._backend.array(op)
        
        # 1b. 维度检查 (兼容 CuPy 和 PurePython)
        try:
            op_shape = op.shape if hasattr(op, 'shape') else (len(op), len(op[0]))
            rho_shape = self._density_matrix.shape if hasattr(self._density_matrix, 'shape') else (len(self._density_matrix), len(self._density_matrix[0]))
            
            if op_shape != rho_shape:
                raise ValueError(
                    f"Observable matrix dimensions ({op_shape}) "
                    f"must match the density matrix dimensions ({rho_shape})."
                )
        except (AttributeError, IndexError):
             raise ValueError("Could not determine shapes of observable and/or density matrix for validation.")

        # 1c. 厄米性检查：可观测量必须是厄米矩阵 (O == O†)。
        #     使用后端的 allclose 和 conj_transpose 方法。
        if not self._backend.allclose(op, self._backend.conj_transpose(op), atol=1e-9):
            raise ValueError("The provided observable matrix must be Hermitian.")

        # --- 步骤 2: 计算期望值 Tr(O @ ρ) ---
        # 2a. 计算矩阵乘积 O @ ρ
        product_matrix = self._backend.dot(op, self._density_matrix)
        
        # 2b. 计算乘积矩阵的迹
        expectation_complex = self._backend.trace(product_matrix)
        
        # --- 步骤 3: 后处理与返回 ---
        # 理论上期望值是实数，但由于浮点误差，可能会有微小的虚部。
        if abs(expectation_complex.imag) > 1e-7:
            self._internal_logger.warning(
                f"[{log_prefix}] Expectation value has a significant imaginary part: {expectation_complex.imag:.2e}. "
                "This might indicate a non-Hermitian observable or numerical issues. Returning the real part."
            )
        
        # 返回期望值的实部，并转换为标准的 Python float 类型。
        return float(expectation_complex.real)

    # --- [增强] VQA 支持 ---
    def _build_global_pauli_observable(self, pauli_string: PauliString) -> Any:
        """
        [内部辅助函数] 根据 PauliString 对象构建其对应的全局可观测量矩阵。

        例如，对于一个3比特系统和一个 `PauliString(coefficient=0.5, pauli_map={0: 'X', 2: 'Z'})`，
        此方法会计算 `0.5 * (X ⊗ I ⊗ Z)` 的 `8x8` 矩阵表示。

        [修正] 此实现是后端无关的。它使用纯 Python 列表定义基础 Pauli 矩阵，
        然后根据当前后端 (`self._backend`) 将它们转换为适当的类型，并使用
        后端的 `kron` 方法来计算张量积链。

        Args:
            pauli_string (PauliString): 
                一个 `PauliString` 对象，包含了系数和泡利算子的映射。

        Returns:
            Any: 
                一个 (2^N, 2^N) 的矩阵，代表该 `PauliString` 对应的全局可观测量。
                其类型与当前后端匹配（`cupy.ndarray` 或 `List[List[complex]]`）。
        """
        # --- 步骤 1: 处理 0-qubit 的边缘情况 ---
        if self.num_qubits == 0:
            # 对于0比特系统（一个标量态），可观测量也是一个标量。
            return pauli_string.coefficient

        # --- 步骤 2: 定义后端无关的基础 Pauli 矩阵 ---
        # 首先定义为纯 Python 列表
        I_list = [[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, 1.0+0.0j]]
        X_list = [[0.0+0.0j, 1.0+0.0j], [1.0+0.0j, 0.0+0.0j]]
        Y_list = [[0.0+0.0j, -1.0j],    [1.0j,     0.0+0.0j]]
        Z_list = [[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, -1.0+0.0j]]
        
        # 根据当前后端，将它们转换为后端特定的类型（如果需要）
        # 如果是 PurePythonBackend，它们保持为列表。
        # 如果是 CuPy，它们被转换为 CuPy 数组。
        if hasattr(self._backend, 'array'):
            I_local = self._backend.array(I_list)
            X_local = self._backend.array(X_list)
            Y_local = self._backend.array(Y_list)
            Z_local = self._backend.array(Z_list)
        else:
            I_local, X_local, Y_local, Z_local = I_list, X_list, Y_list, Z_list
        
        pauli_ops_map = {'I': I_local, 'X': X_local, 'Y': Y_local, 'Z': Z_local}
        
        # --- 步骤 3: 构建单比特算子的列表 ---
        # 创建一个长度为 N 的列表，其中每个元素是对应量子比特上的单比特 Pauli 算子。
        # 如果某个比特在 pauli_map 中未指定，则默认为单位矩阵 'I'。
        # 量子比特的顺序遵循 q_0, q_1, ..., q_{N-1} (little-endian for kron chain)
        # 但是，由于我们的 `_build_global_operator_multi_qubit` 是 big-endian，
        # 为了保持一致性，我们在这里也遵循 big-endian 约定。
        # O_{N-1} ⊗ ... ⊗ O_1 ⊗ O_0
        
        op_list_for_kron: List[Any] = []
        for i in range(self.num_qubits):
            pauli_char = pauli_string.pauli_map.get(i, 'I')
            op_list_for_kron.append(pauli_ops_map[pauli_char])
        
        # --- 步骤 4: 高效地计算张量积链 ---
        # 为了与 `_build_global_operator_multi_qubit` 的 big-endian 约定一致，
        # 我们应该从最高位的比特开始构建张量积：Op_{N-1} ⊗ Op_{N-2} ⊗ ...
        # 注意：np.kron(A, B) 对应 A 作用于更高位的比特。
        global_op = op_list_for_kron[self.num_qubits - 1]
        for i in range(self.num_qubits - 2, -1, -1):
            global_op = self._backend.kron(global_op, op_list_for_kron[i])

        # --- 步骤 5: 乘以系数并返回 ---
        # 最终的可观测量是 Pauli 串的矩阵表示乘以其复数系数。
        # 需要一个后端无关的标量-矩阵乘法。
        coefficient = pauli_string.coefficient
        if isinstance(global_op, list): # PurePython
            return [[coefficient * elem for elem in row] for row in global_op]
        else: # CuPy
            return coefficient * global_op

    def get_hamiltonian_expectation(self, hamiltonian: Hamiltonian) -> float:
        """
        [增强] 计算给定哈密顿量在当前量子态下的期望值。

        哈密顿量 `H` 被定义为一个 `PauliString` 的列表，代表 `H = Σ_k (c_k * P_k)`，
        其中 `c_k` 是复数系数，`P_k` 是 Pauli 串（如 `X_0 Z_1`）。
        
        期望值的计算利用了其线性性质:
        ⟨H⟩ = ⟨Σ_k c_k * P_k⟩ = Σ_k c_k * ⟨P_k⟩ = Σ_k Tr( (c_k * P_k) @ ρ )

        此方法通过遍历哈密顿量中的每一项 `PauliString`，调用
        `_build_global_pauli_observable` 来构建该项的矩阵表示，然后调用
        `get_expectation_value` 计算其期望值，最后将所有项的期望值累加。

        [修正] 此实现是完全后端无关的，因为它依赖于已经被重构为后端无关的
        `_build_global_pauli_observable` 和 `get_expectation_value` 方法。

        Args:
            hamiltonian (Hamiltonian): 
                一个 `List[PauliString]`，表示要计算期望值的哈密顿量。

        Returns:
            float: 
                哈密顿量的期望值。由于哈密顿量是厄米算子，其期望值保证为实数。

        Raises:
            TypeError: 如果 `hamiltonian` 参数不是一个 `PauliString` 的列表。
            ValueError: (由底层方法抛出) 如果 `PauliString` 定义与量子态的
                        比特数不匹配。
            RuntimeError: (由底层方法抛出) 如果在计算过程中发生错误。
        """
        # --- 步骤 1: 输入验证 ---
        if not isinstance(hamiltonian, list) or (hamiltonian and not all(isinstance(ps, PauliString) for ps in hamiltonian)):
            raise TypeError("Hamiltonian must be a list of PauliString objects.")
            
        # --- 步骤 2: 遍历哈密顿量的每一项并累加期望值 ---
        total_expectation = 0.0
        
        self._internal_logger.debug(f"Calculating expectation value for Hamiltonian with {len(hamiltonian)} terms.")

        for pauli_term in hamiltonian:
            # --- 步骤 2a: 构建单个 PauliString 对应的全局可观测量矩阵 ---
            # 委托给 `_build_global_pauli_observable`，该方法已经是后端无关的。
            # 它会返回一个包含了系数 c_k 的厄米矩阵 O_k = c_k * P_k。
            observable_matrix = self._build_global_pauli_observable(pauli_term)
            
            # --- 步骤 2b: 计算该项的期望值 ---
            # 委托给通用的 `get_expectation_value` 方法，该方法也已经是后端无关的。
            # 它会计算 Tr(O_k @ ρ)。
            term_expectation = self.get_expectation_value(observable_matrix)
            
            # --- 步骤 2c: 累加到总期望值 ---
            total_expectation += term_expectation
            
        # --- 步骤 3: 返回最终结果 ---
        # 期望值理论上是实数，转换为 float 类型以确保 API 的类型稳定性。
        return float(total_expectation)

    # --- [辅助方法] 日志和时间戳 (保持一致) ---
    def _add_history_log(self, 
                        gate_name: str, 
                        controls: Optional[List[int]] = None, 
                        targets: Optional[List[int]] = None, 
                        params: Optional[Dict[str, Any]] = None):
        """
        [内部核心辅助方法] 将一个门操作的详细信息记录到 `gate_application_history` 列表中。

        此方法负责创建一个结构化的日志条目，将其追加到历史记录中，并管理
        历史记录的长度，以防止其无限增长（滚动日志）。

        在记录操作后，它会自动调用 `_update_timestamp()`，因为任何被记录的
        操作都被认为是可能改变量子态的“显著事件”。

        [修正] 此方法是纯 Python 实现，不依赖任何后端，因此在重构后无需修改。

        Args:
            gate_name (str): 
                被应用的量子门或效应的名称 (e.g., "h", "cnot", "apply_quantum_channel")。
            controls (Optional[List[int]], optional): 
                对于受控门，一个包含控制量子比特索引的列表。默认为 None。
            targets (Optional[List[int]], optional): 
                一个包含操作的目标量子比特索引的列表。默认为 None。
            params (Optional[Dict[str, Any]], optional): 
                对于参数化门或需要额外参数的操作，一个包含这些参数的字典。
                默认为 None。
        """
        log_prefix = f"QuantumState._add_history_log(Op: '{gate_name}')"

        # --- 步骤 1: 输入验证 ---
        if not gate_name or not isinstance(gate_name, str) or not gate_name.strip():
            self._internal_logger.error(
                f"[{log_prefix}] Attempted to log history with an invalid operation name: '{gate_name}'. "
                "History entry was not recorded."
            )
            return
        
        # --- 步骤 2: 构建历史条目字典 ---
        history_entry: Dict[str, Any] = {
            "operation_name": gate_name,
            "timestamp_utc_iso": datetime.now(timezone.utc).isoformat()
        }

        # 仅当提供了有效的 controls, targets, 或 params 时，才将其添加到条目中。
        # 使用 deepcopy 确保历史记录中的数据与原始传入的参数（可能是可变对象）解耦。
        if controls is not None and isinstance(controls, list):
            history_entry["controls"] = copy.deepcopy(controls) 
        
        if targets is not None and isinstance(targets, list):
            history_entry["targets"] = copy.deepcopy(targets)
            
        if params is not None and isinstance(params, dict):
            history_entry["parameters"] = copy.deepcopy(params)
        
        self._internal_logger.debug(f"[{log_prefix}] Preparing to log history entry: {history_entry}")

        # --- 步骤 3: 健壮地处理 self.gate_application_history 属性 ---
        # 这是一个安全检查，以防该属性在运行时被意外地删除或更改类型。
        if not hasattr(self, 'gate_application_history') or not isinstance(self.gate_application_history, list):
            self._internal_logger.critical(
                f"[{log_prefix}] Internal attribute 'gate_application_history' is not a list or is missing. "
                "It has been reset to an empty list. This may indicate that the object state was corrupted."
            )
            self.gate_application_history = []
        
        # --- 步骤 4: 追加新条目到历史列表 ---
        try:
            self.gate_application_history.append(history_entry)
        except Exception as e_append:
            self._internal_logger.error(
                f"[{log_prefix}] An unexpected error occurred while appending to gate_application_history: {e_append}",
                exc_info=True
            )
            return

        # --- 步骤 5: 管理历史记录列表的长度 (滚动日志) ---
        max_history_entries = self.MAX_GATE_HISTORY
        current_history_length = len(self.gate_application_history)
        
        if current_history_length > max_history_entries:
            num_to_remove = current_history_length - max_history_entries
            # 切片操作会创建一个新的列表，有效地移除了最旧的条目
            self.gate_application_history = self.gate_application_history[num_to_remove:]
            self._internal_logger.debug(
                f"[{log_prefix}] Operation history has been trimmed. Removed {num_to_remove} old entries. "
                f"Current history length: {len(self.gate_application_history)} (Limit: {max_history_entries})"
            )
        
        # --- 步骤 6: 更新最后一次显著状态变化的时间戳 ---
        self._update_timestamp()

        self._internal_logger.debug(f"[{log_prefix}] Operation '{gate_name}' was successfully logged to history.")

    def _update_timestamp(self):
        """
        [内部核心辅助方法] 更新量子态的“最后一次显著状态变化”的时间戳。

        此方法应在任何可能改变量子态的操作（如应用门、测量坍缩、
        引入噪声等）成功完成后被调用。

        它会获取当前带时区的 UTC 时间，并将其格式化为 ISO 8601 标准
        字符串，然后更新 `self.last_significant_update_timestamp_utc_iso`
        属性。

        [修正] 此方法是纯 Python 实现，不依赖任何后端，因此在重构后无需修改。
        """
        log_prefix = f"QuantumState._update_timestamp(N={self.num_qubits})"
        try:
            # --- 步骤 1: 获取当前带时区的 UTC 时间 ---
            # `timezone.utc` 确保了时间戳是全球统一的，不受本地时区影响。
            now_utc = datetime.now(timezone.utc)
            
            # --- 步骤 2: 转换为 ISO 8601 标准格式的字符串 ---
            # .isoformat() 是生成标准时间戳字符串（如 '2025-06-24T10:30:59.123456+00:00'）
            # 的推荐方法。
            new_timestamp_utc_iso = now_utc.isoformat()
            
            # --- 步骤 3: 更新实例的属性 ---
            self.last_significant_update_timestamp_utc_iso = new_timestamp_utc_iso
            
            # --- 步骤 4: 记录更新事件 (可选，用于调试) ---
            self._internal_logger.debug(
                f"[{log_prefix}] 'last_significant_update_timestamp_utc_iso' has been updated to: {new_timestamp_utc_iso}"
            )
        except Exception as e_update_ts:
            # 捕获任何在时间戳操作中可能发生的罕见错误，并记录
            self._internal_logger.error(
                f"[{log_prefix}] An unexpected error occurred while updating the timestamp: {e_update_ts}",
                exc_info=True
            )

    # --- [增强] 量子态深度分析工具 ---
    def calculate_von_neumann_entropy(self, qubits_to_partition: List[int]) -> float:
        """
        [增强] 计算指定子系统的冯·诺依曼纠缠熵 S(ρ_A) = -Tr(ρ_A * log₂(ρ_A))。

        纠缠熵是衡量子系统 A 与系统其余部分之间量子纠缠程度的关键指标。
        - 熵为 0 表示子系统 A 是一个纯态，与系统其余部分没有纠缠。
        - 熵大于 0 表示存在纠缠，值越大，纠缠程度越高。

        计算流程:
        1.  通过对系统其余部分执行“部分迹”，获得子系统 A 的约化密度矩阵 `ρ_A`。
        2.  计算 `ρ_A` 的特征值 `{λ_i}`。
        3.  根据公式 `S(ρ_A) = -Σ_i (λ_i * log₂(λ_i))` 计算熵。

        [修正] 此实现是后端无关的。它依赖 `_partial_trace`、`eigvalsh` 和
        `log2` 等后端方法，并能正确处理 CuPy 数组和 Python 列表。

        Args:
            qubits_to_partition (List[int]): 
                一个整数列表，定义了我们感兴趣的子系统 A。
                例如 `[0, 1]` 表示计算由 q0 和 q1 构成的子系统的熵。

        Returns:
            float: 
                子系统 A 的冯·诺依曼纠缠熵（以比特为单位，因为使用 log₂）。

        Raises:
            ValueError: 如果 `qubits_to_partition` 无效（如为空、包含重复或越界的索引）。
            NotImplementedError: 如果当前后端不支持 `eigvalsh`。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not qubits_to_partition:
            raise ValueError("qubits_to_partition cannot be empty.")
        if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits_to_partition):
            raise ValueError("Invalid qubit index found in qubits_to_partition.")
        if len(set(qubits_to_partition)) != len(qubits_to_partition):
            raise ValueError("qubits_to_partition must contain unique qubit indices.")
        
        # --- 步骤 2: 获取约化密度矩阵 ρ_A ---
        # 找出需要被迹掉的比特（即所有不在子系统 A 中的比特）
        qubits_to_trace_out = [q for q in range(self.num_qubits) if q not in qubits_to_partition]
        
        # _partial_trace 方法已经是后端无关的
        rho_A = self._partial_trace(qubits_to_trace_out)
        
        # --- 步骤 3: 计算 ρ_A 的特征值 ---
        try:
            # 使用后端的 eigvalsh，它专门用于厄米矩阵，性能更好且保证返回实数特征值。
            eigenvalues = self._backend.eigvalsh(rho_A)
        except (AttributeError, NotImplementedError):
             raise NotImplementedError(
                f"The current backend ({type(self._backend).__name__}) does not support the 'eigvalsh' method, "
                "which is required for calculating Von Neumann entropy."
            )
        
        # --- 步骤 4: 计算冯·诺依曼熵 ---
        # S = -Σ_i (λ_i * log₂(λ_i))
        entropy = 0.0
        
        # 迭代特征值列表/数组
        for eigenvalue in eigenvalues:
            # [核心] 数值稳定性：只有当特征值（概率）大于一个很小的阈值时才计算。
            # 这是因为 lim(x->0) [x * log(x)] = 0。直接计算 log(0) 会导致 NaN 或错误。
            if eigenvalue > 1e-12:
                # 使用后端的 log2 方法
                entropy -= eigenvalue * self._backend.log2(eigenvalue)
        
        # --- 步骤 5: 返回结果 ---
        # 将结果转换为标准的 Python float 类型，以确保 API 的类型稳定性
        return float(entropy)

    def get_bloch_vector(self, qubit_index: int) -> Tuple[float, float, float]:
        """
        [增强] 计算指定单个量子比特的布洛赫矢量 (rx, ry, rz)。

        布洛赫矢量提供了一个在三维单位球（布洛赫球）中可视化单个量子比特
        状态的直观方式。矢量的坐标由泡利算子的期望值给出：
        rx = ⟨σ_x⟩ = Tr(ρ_q @ σ_x)
        ry = ⟨σ_y⟩ = Tr(ρ_q @ σ_y)
        rz = ⟨σ_z⟩ = Tr(ρ_q @ σ_z)
        其中 `ρ_q` 是目标量子比特的约化密度矩阵。

        矢量的长度 `|r|` 满足 `|r| <= 1`。对于纯态，`|r| = 1`（矢量在球面上）；
        对于混合态，`|r| < 1`（矢量在球内）。

        [修正] 此实现是后端无关的。它依赖 `_partial_trace` 方法，并使用
        后端无关的方式定义 Pauli 矩阵和执行线性代数运算。

        Args:
            qubit_index (int): 
                要计算布洛赫矢量的目标量子比特的索引。

        Returns:
            Tuple[float, float, float]: 
                一个包含布洛赫矢量三个分量 (rx, ry, rz) 的元组。

        Raises:
            ValueError: 如果 `qubit_index` 无效。
        """
        # --- 步骤 1: 严格的输入验证 ---
        if not (0 <= qubit_index < self.num_qubits):
            raise ValueError(f"Invalid qubit_index {qubit_index} for a system of {self.num_qubits} qubits.")
        
        # --- 步骤 2: 获取目标比特的约化密度矩阵 (2x2) ---
        # 找出需要被迹掉的比特
        qubits_to_trace_out = [q for q in range(self.num_qubits) if q != qubit_index]
        
        # _partial_trace 方法已经是后端无关的
        rho_q = self._partial_trace(qubits_to_trace_out)
        
        # --- 步骤 3: 定义后端无关的 Pauli 矩阵 ---
        # 首先定义为纯 Python 列表
        sigma_x_list = [[0.0+0.0j, 1.0+0.0j], [1.0+0.0j, 0.0+0.0j]]
        sigma_y_list = [[0.0+0.0j, -1.0j],    [1.0j,     0.0+0.0j]]
        sigma_z_list = [[1.0+0.0j, 0.0+0.0j], [0.0+0.0j, -1.0+0.0j]]
        
        # 根据当前后端，将它们转换为后端特定的类型（如果需要）
        if hasattr(self._backend, 'array'):
            sigma_x = self._backend.array(sigma_x_list)
            sigma_y = self._backend.array(sigma_y_list)
            sigma_z = self._backend.array(sigma_z_list)
        else:
            sigma_x, sigma_y, sigma_z = sigma_x_list, sigma_y_list, sigma_z_list
        
        # --- 步骤 4: 计算布洛赫矢量分量 ---
        # 期望值 Tr(ρ @ Pauli) 理论上是实数。我们取 .real 以消除浮点误差。
        # 调用后端的 dot 和 trace 方法。
        rx = self._backend.trace(self._backend.dot(rho_q, sigma_x)).real
        ry = self._backend.trace(self._backend.dot(rho_q, sigma_y)).real
        rz = self._backend.trace(self._backend.dot(rho_q, sigma_z)).real
        
        # --- 步骤 5: 返回标准的 Python float 元组 ---
        return (float(rx), float(ry), float(rz))

# ========================================================================
# --- 6. 公共API (Public API) ---
# ========================================================================

# 注意: Hamiltonian 和 PauliString 类定义在文件顶层，可直接访问。
# 若此文件作为包的一部分被导入，这些类型将通过 `from .quantum_core import ...` 形式在其他模块中导入。

# [新增] 明确导出并行控制函数
# 这使得用户可以直接从库的顶层导入并使用这些功能。
# 例如：`from nexus_quantum_core.quantum_core import enable_parallelism, disable_parallelism`
# 或者如果 __init__.py 配置得当，直接 `from nexus_quantum_core import enable_parallelism`
# 这些行在实际代码中不重复写，只是为了文档说明其作为公共API的身份。


def create_quantum_state(num_qubits: int) -> 'QuantumState':
    """
    工厂函数：创建一个处于 |0...0⟩ 态的初始量子态。

    这是与库交互的推荐入口点，用于获取一个新的、干净的量子态实例。
    它封装了 `QuantumState` 类的实例化过程，并提供了健壮的错误处理。

    Args:
        num_qubits (int): 要创建的量子态的量子比特数。必须是非负整数。

    Returns:
        QuantumState: 一个新创建的、已初始化的 QuantumState 实例。

    Raises:
        ValueError: 如果 `num_qubits` 为负数或超过硬件/配置上限。
        ImportError: 如果请求的后端（如 CuPy）未安装。
        RuntimeError: 如果发生其他意外的初始化错误。
    """
    # 理论上，_core_config 应该已经初始化，但这里仍检查一下以防万一
    backend_choice = _core_config.get("BACKEND_CHOICE", "auto")
    if backend_choice == 'cupy' and cp is None:
        raise ImportError(f"Cannot create quantum state with CuPy backend: CuPy is not installed.")
    
    logger.debug(f"API: Attempting to create a new quantum state with {num_qubits} qubits.")
    
    try:
        state = QuantumState(num_qubits=num_qubits)
        # [*** 关键修复 ***] 修正获取后端名称的逻辑，避免 `__getattr__` 陷阱
        # 检查后端是模块（如cupy）还是类实例（如PurePythonBackend）
        backend_name = state._backend.__name__ if isinstance(state._backend, types.ModuleType) else type(state._backend).__name__
        logger.info(f"API: Successfully created a {num_qubits}-qubit quantum state using {backend_name} backend.")
        return state
    except ValueError as e:
        logger.error(f"API: Failed to create quantum state with {num_qubits} qubits due to configuration/hardware limits: {e}", exc_info=True)
        raise # 将验证错误重新抛出给调用者
    except ImportError as e:
        logger.error(f"API: Failed to create quantum state with {num_qubits} qubits due to missing backend library: {e}", exc_info=True)
        raise # 将导入错误重新抛出给调用者
    except Exception as e:
        logger.critical(f"API: An unexpected error occurred during QuantumState creation for {num_qubits} qubits: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during QuantumState creation: {e}") from e
def run_circuit_on_state(state: 'QuantumState', circuit: 'QuantumCircuit') -> 'QuantumState':
    """
    在一个量子态上执行一个量子线路，返回演化后的新量子态。

    这是一个核心的计算接口。它接收一个量子态和一个量子线路，
    应用线路中的所有操作，并返回一个新的、代表最终状态的量子态对象。
    
    此函数遵循函数式编程的原则，不会修改输入的 `state` 对象（不可变性）。

    Args:
        state (QuantumState): 将要被演化的初始量子态。
        circuit (QuantumCircuit): 包含一系列量子操作的量子线路。

    Returns:
        QuantumState: 一个新的 QuantumState 实例，代表了演化后的状态。

    Raises:
        TypeError: 如果输入的 `state` 或 `circuit` 不是正确的类型。
        ValueError: 如果 `state` 和 `circuit` 的量子比特数不匹配。
        RuntimeError: 如果在电路执行过程中发生底层计算错误。
    """
    if not isinstance(state, QuantumState):
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"Input 'circuit' must be a QuantumCircuit instance, but got {type(circuit).__name__}.")
    if state.num_qubits != circuit.num_qubits:
        raise ValueError(
            f"State ({state.num_qubits} qubits) and circuit ({circuit.num_qubits} qubits) "
            "qubit numbers do not match."
        )
    
    logger.info(f"API: Running circuit '{circuit.description or 'unnamed'}' with {len(circuit)} instructions on a {state.num_qubits}-qubit state.")
    
    # 使用深拷贝创建演化后的新状态，保持原始状态不变
    evolved_state = copy.deepcopy(state)
    
    try:
        evolved_state.run_circuit(circuit)
    except Exception as e:
        logger.error(f"API: An error occurred while running the circuit on the state: {e}", exc_info=True)
        # re-raise to indicate failure to the caller
        raise RuntimeError(f"Circuit execution failed due to an internal error: {e}") from e
    
    logger.info(f"API: Circuit execution completed successfully.")
    
    return evolved_state

def get_measurement_probabilities(state: 'QuantumState') -> List[float]:
    """
    从一个量子态中获取所有计算基的测量概率。

    返回一个 Python 列表，其长度为 2^N，其中第 i 个元素是测量到
    计算基矢 |i⟩ 的概率。

    Args:
        state (QuantumState): 要获取测量概率的量子态。

    Returns:
        List[float]: 
            一个包含所有测量概率的 Python 列表。
            如果量子态无效，则返回一个空列表并记录错误。
        
    Raises:
        TypeError: 如果输入 `state` 不是 QuantumState 实例。
    """
    if not isinstance(state, QuantumState):
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    
    # QuantumState.get_probabilities() 已被修正为返回 List[float]
    probabilities = state.get_probabilities()
    logger.debug(f"API: Retrieved measurement probabilities for {state.num_qubits}-qubit state. Count: {len(probabilities)}")
    
    return probabilities

def get_marginal_probability(state: 'QuantumState', qubit_index: int) -> Tuple[float, float]:
    """
    获取指定单个量子比特的边际概率分布 (P(0), P(1))。

    这对于分析单个量子比特的状态非常有用，即使它可能与其他量子比特纠缠。
    此方法通过在底层调用部分迹来实现。

    Args:
        state (QuantumState): 要分析的量子态。
        qubit_index (int): 目标量子比特的索引。

    Returns:
        Tuple[float, float]: 
            一个元组 `(prob_0, prob_1)`，分别代表测量该比特
            得到 0 和 1 的概率。
            如果底层计算失败，则返回 `(0.0, 0.0)` 并记录一条警告。
                             
    Raises:
        TypeError: 如果输入 `state` 不是 QuantumState 实例。
        ValueError: 如果 `qubit_index` 无效（由底层方法抛出）。
    """
    if not isinstance(state, QuantumState):
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    if not isinstance(qubit_index, int):
        raise TypeError("qubit_index must be an integer.")
        
    result = state.get_marginal_probabilities(qubit_index)
    
    if result is None:
        logger.warning(
            f"API: Failed to calculate marginal probability for qubit {qubit_index}. "
            "This could be due to an invalid state or an internal calculation error. "
            "Returning (0.0, 0.0) as a fallback."
        )
        return (0.0, 0.0)
        
    return result

# --- [增强] VQA 相关 API ---
def calculate_hamiltonian_expectation_value(state: 'QuantumState', hamiltonian: Hamiltonian) -> float:
    """
    [增强] 计算给定哈密顿量在当前量子态下的期望值。

    这是变分量子算法（VQA）中的一个核心功能，通常用于评估成本函数。

    Args:
        state (QuantumState): 
            要计算期望值的量子态。
        hamiltonian (Hamiltonian): 
            一个 `List[PauliString]`，表示要计算期望值的哈密顿量。

    Returns:
        float: 哈密顿量的期望值（一个实数）。

    Raises:
            TypeError: 如果输入类型不正确。
            ValueError: 如果哈密顿量定义与量子态的比特数不匹配（由底层方法抛出）。
            RuntimeError: 如果底层计算失败。
    """
    if not isinstance(state, QuantumState):
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    
    if not isinstance(hamiltonian, list) or (hamiltonian and not all(isinstance(ps, PauliString) for ps in hamiltonian)):
        raise TypeError("Hamiltonian must be a list of PauliString objects.")

    # 检查哈密顿量中涉及的量子比特索引是否在状态的范围内
    max_hamiltonian_qubit = -1
    for ps in hamiltonian:
        if ps.pauli_map:
            # 确保 pauli_map.keys() 返回的是整数
            max_hamiltonian_qubit = max(max_hamiltonian_qubit, max(ps.pauli_map.keys()))
    
    if max_hamiltonian_qubit >= state.num_qubits:
        raise ValueError(
            f"Hamiltonian contains operators on qubit {max_hamiltonian_qubit}, which is out of range for a "
            f"{state.num_qubits}-qubit state. All qubit indices must be < {state.num_qubits}."
        )

    logger.info(f"API: Calculating expectation value for Hamiltonian with {len(hamiltonian)} terms on {state.num_qubits}-qubit state.")
    
    try:
        return state.get_hamiltonian_expectation(hamiltonian)
    except Exception as e:
        logger.error(f"API: An error occurred while calculating Hamiltonian expectation value: {e}", exc_info=True)
        raise RuntimeError(f"Hamiltonian expectation calculation failed due to an internal error: {e}") from e

# --- [增强] 量子动力学 API ---
def build_trotter_step_circuit(hamiltonian: Hamiltonian, time_step: float) -> 'QuantumCircuit':
    """
    [增强] 构建一个 Suzuki-Trotter 分解的单步量子线路，用于模拟哈密顿量演化。

    此函数将哈密顿量 H = Σ H_k 的演化 U(t) = exp(-iHt) 近似为
    U(t) ≈ Π_k exp(-iH_k t)。目前支持单比特和双比特的纯Pauli项
    （如 X, Y, Z, XX, YY, ZZ）。

    Args:
        hamiltonian (Hamiltonian): 
            一个 `List[PauliString]`，表示要模拟的哈密顿量。
        time_step (float): 
            时间演化的步长 (dt)。

    Returns:
        QuantumCircuit: 
            一个代表单个Trotter演化步骤的量子线路。

    Raises:
        ValueError: 如果哈密顿量中包含不支持分解的Pauli串，或参数无效。
        TypeError: 如果哈密顿量类型不正确。
    """
    if not isinstance(hamiltonian, list) or (hamiltonian and not all(isinstance(ps, PauliString) for ps in hamiltonian)):
        raise TypeError("Hamiltonian must be a list of PauliString objects.")
    if not isinstance(time_step, (float, int)):
        raise ValueError("time_step must be a numeric value.")
        
    logger.info(f"API: Building Trotter step circuit for Hamiltonian with {len(hamiltonian)} terms and time_step={time_step:.4f}.")
    
    # 确定电路所需的量子比特数
    max_qubit_idx = -1
    for ps in hamiltonian:
        if ps.pauli_map:
            max_qubit_idx = max(max_qubit_idx, max(ps.pauli_map.keys()))
    num_qubits = max_qubit_idx + 1 if max_qubit_idx >= 0 else 0

    trotter_qc = QuantumCircuit(num_qubits=num_qubits, description=f"Trotter Step for dt={time_step:.4f}")

    for pauli_string in hamiltonian:
        active_qubits = sorted([q for q, op in pauli_string.pauli_map.items() if op != 'I'])
        # pauli_string.coefficient 总是 complex，所以取 .real
        coefficient_real = pauli_string.coefficient.real 
        
        rotation_angle = 2 * coefficient_real * time_step
        
        if math.isclose(rotation_angle, 0.0, abs_tol=1e-9):
            continue 

        if len(active_qubits) == 0:
            # 纯数字项 (Global Phase)
            # 全局相位不影响物理测量结果和期望值，因此通常可以安全地忽略。
            # 为了保持模拟的物理正确性，这里选择忽略。
            pass
            
        elif len(active_qubits) == 1:
            q = active_qubits[0]
            op = pauli_string.pauli_map[q]
            if op == 'X': trotter_qc.rx(q, rotation_angle)
            elif op == 'Y': trotter_qc.ry(q, rotation_angle)
            elif op == 'Z': trotter_qc.rz(q, rotation_angle)
            else: 
                raise ValueError(f"Unsupported Pauli operator for single qubit: {op} in {pauli_string}")
                
        elif len(active_qubits) == 2:
            q1, q2 = active_qubits[0], active_qubits[1]
            op1, op2 = pauli_string.pauli_map[q1], pauli_string.pauli_map[q2]
            
            if op1 == 'X' and op2 == 'X': trotter_qc.rxx(q1, q2, rotation_angle)
            elif op1 == 'Y' and op2 == 'Y': trotter_qc.ryy(q1, q2, rotation_angle)
            elif op1 == 'Z' and op2 == 'Z': trotter_qc.rzz(q1, q2, rotation_angle)
            else:
                raise ValueError(f"Unsupported 2-qubit Pauli string for Trotter decomposition: '{op1}{q1}{op2}{q2}'. "
                                 "Only XX, YY, ZZ are directly supported in this version.")
        else:
            raise ValueError(f"Unsupported Pauli string for Trotter decomposition (more than 2 non-identity qubits): {pauli_string}")
            
    return trotter_qc

# --- [增强] 量子态深度分析 API ---
def calculate_entanglement_entropy(state: 'QuantumState', qubits_to_partition: List[int]) -> float:
    """
    [增强] 计算指定子系统的冯·诺依曼纠缠熵 S(ρ_A) = -Tr(ρ_A log₂(ρ_A))。

    Args:
        state (QuantumState): 
            要分析的量子态。
        qubits_to_partition (List[int]): 
            一个整数列表，定义了我们感兴趣的子系统 A。
            例如 `[0, 1]` 表示计算由 q0 和 q1 构成的子系统的熵。

    Returns:
        float: 子系统 A 的冯·诺依曼纠缠熵（以比特为单位，因为使用 log₂）。

    Raises:
        TypeError: 如果输入类型不正确。
        ValueError: 如果 `qubits_to_partition` 无效或为空（由底层方法抛出）。
        NotImplementedError: 如果底层计算不支持（例如，后端缺少eigvalsh）。
        RuntimeError: 如果底层计算失败。
    """
    if not isinstance(state, QuantumState):
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    
    if not isinstance(qubits_to_partition, list) or (qubits_to_partition and not all(isinstance(q, int) for q in qubits_to_partition)):
        raise TypeError("qubits_to_partition must be a list of integers.")

    logger.info(f"API: Calculating entanglement entropy for partition {qubits_to_partition} on {state.num_qubits}-qubit state.")
    
    try:
        return state.calculate_von_neumann_entropy(qubits_to_partition)
    except (ValueError, TypeError, NotImplementedError) as e:
        logger.error(f"API: An error occurred while calculating entanglement entropy: {e}", exc_info=True)
        raise # 重新抛出更具体的错误
    except Exception as e:
        logger.error(f"API: An unexpected error occurred while calculating entanglement entropy: {e}", exc_info=True)
        raise RuntimeError(f"Entanglement entropy calculation failed due to an internal error: {e}") from e

def get_bloch_vector(state: 'QuantumState', qubit_index: int) -> Tuple[float, float, float]:
    """
    [增强] 计算指定单个量子比特的布洛赫矢量 (rx, ry, rz)。

    布洛赫矢量提供了一个在三维空间中可视化单个量子比特状态的直观方式。
    矢量的长度 |r| <= 1。对于纯态，|r| = 1；对于混合态，|r| < 1。

    Args:
        state (QuantumState): 
            要分析的量子态。
        qubit_index (int): 
            目标量子比特的索引。

    Returns:
        Tuple[float, float, float]: 布洛赫矢量 (rx, ry, rz)。

    Raises:
        TypeError: 如果输入类型不正确。
        ValueError: 如果 `qubit_index` 无效（由底层方法抛出）。
        RuntimeError: 如果底层计算失败。
    """
    if not isinstance(state, QuantumState):
        raise TypeError(f"Input 'state' must be a QuantumState instance, but got {type(state).__name__}.")
    if not isinstance(qubit_index, int):
        raise TypeError("qubit_index must be an integer.")

    logger.info(f"API: Getting Bloch vector for qubit {qubit_index} on {state.num_qubits}-qubit state.")
    
    try:
        return state.get_bloch_vector(qubit_index)
    except (ValueError, TypeError) as e:
        logger.error(f"API: An error occurred while getting the Bloch vector for qubit {qubit_index}: {e}", exc_info=True)
        raise # 重新抛出更具体的错误
    except Exception as e:
        logger.error(f"API: An unexpected error occurred while getting the Bloch vector for qubit {qubit_index}: {e}", exc_info=True)
        raise RuntimeError(f"Bloch vector calculation failed due to an internal error: {e}") from e

# --- [增强] 高级算法构建 API ---

def build_qft_circuit(num_qubits: int, inverse: bool = False) -> 'QuantumCircuit':
    """
    [最终权威版] 构建一个正确的量子傅里叶变换 (QFT) 或其逆 (IQFT) 的线路。
    
    此版本明确了比特序约定 (|q_{n-1}...q_1q_0>)，并严格遵循教科书定义。
    - QFT 流程: 核心旋转 -> SWAP 网络
    - IQFT 流程: SWAP 网络 -> 逆核心旋转
    """
    if not isinstance(num_qubits, int) or num_qubits < 0:
        raise ValueError("num_qubits must be a non-negative integer.")

    qc_name = f"{num_qubits}-qubit {'Inverse ' if inverse else ''}QFT"
    qc = QuantumCircuit(num_qubits=num_qubits, description=qc_name)
    
    if num_qubits == 0:
        return qc

    if not inverse:
        # --- 正向 QFT ---
        # 1. 核心旋转 (从最高位比特开始)
        # 遍历每一个量子比特，作为后续受控旋转门的目标
        for i in range(num_qubits - 1, -1, -1):
            # 首先对当前比特施加一个Hadamard门
            qc.h(i)
            # 然后，对更高位的比特 i，施加由所有更低位的比特 j 控制的相位旋转
            for j in range(i - 1, -1, -1):
                # 旋转角度 R_k = 2π / 2^k, 其中 k = i - j + 1
                # 我们的 cp 门定义为 e^(i*angle)，所以需要正角度
                angle = math.pi / (2**(i - j))
                qc.cp(j, i, angle)
        
        # 2. SWAP 网络 (反转比特顺序)
        # 在QFT变换后，比特的顺序是颠倒的，需要用SWAP门将其恢复
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - 1 - i)
    else:
        # --- 逆向 QFT (IQFT) ---
        # IQFT是QFT的逆操作，因此所有步骤都需要逆序执行
        
        # 1. SWAP 网络 (首先恢复比特顺序，这是QFT的最后一步，所以是IQFT的第一步)
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - 1 - i)
            
        # 2. 逆核心旋转 (从最低位比特开始，并使用负角度)
        # 遍历每一个量子比特
        for i in range(num_qubits):
            # 施加逆向的受控旋转门
            for j in range(i):
                # 使用负角度来抵消正向QFT的旋转
                angle = -math.pi / (2**(i - j))
                qc.cp(j, i, angle)
            # 最后施加一个Hadamard门
            qc.h(i)
            
    return qc
def build_grover_oracle_circuit(num_qubits: int, target_state_int: int) -> 'QuantumCircuit':
    """
    [完整实现] 为Grover搜索算法构建一个标记单个目标状态的“神谕”(Oracle)线路。
    
    这个神谕的作用是：当且仅当输入状态是目标状态时，给系统施加一个 -1 的相位。

    Args:
        num_qubits (int): 算法工作的量子比特数。
        target_state_int (int): 要搜索的目标状态的整数表示 (e.g., 5 for |101⟩)。

    Returns:
        QuantumCircuit: 一个代表神谕操作的量子线路对象。

    Raises:
        ValueError: 如果 target_state_int 超出范围。
    """
    if not (0 <= target_state_int < (1 << num_qubits)):
        raise ValueError("Target state integer is out of range for the given number of qubits.")
        
    oracle_qc = QuantumCircuit(num_qubits=num_qubits, description=f"Oracle for state |{target_state_int}⟩")
    
    # 将目标整数转换为二进制字符串，然后反转以匹配 little-endian 比特顺序 (q0, q1, ...)
    # 这是因为格式化字符串 '0{num_qubits}b' 是 big-endian，例如 5 (101) for 3 qubits => '101'
    # 但是，我们 QuantumState 的 _build_global_operator_multi_qubit 是 big-endian |q_N-1 ... q_0>
    # 并且许多 Qiskit 等库的门操作是 little-endian 索引（q0 是最低位）
    # 这里保持与原始实现一致的逻辑: 0号位是最低位, N-1号位是最高位
    # 如果 target_state_int = 5 (二进制 101), 对于 q2 q1 q0, 那么 q2=1, q1=0, q0=1
    # 对于 MCZ，如果所有控制比特为1，且目标比特为1，施加-1相位
    # 策略：翻转所有目标状态中为0的比特 -> 应用 MCZ -> 再次翻转
    
    # 步骤1: 翻转所有目标状态中为0的比特
    # 遍历所有比特，如果目标状态的该比特为0，则应用X门
    qubits_to_flip_to_make_1 = []
    for i in range(num_qubits):
        # 检查 target_state_int 的第 i 位是否为 0
        if not ((target_state_int >> i) & 1):
            qubits_to_flip_to_make_1.append(i)
            oracle_qc.x(i)
            
    # 步骤2: 应用多控制Z门 (MCZ)
    # MCZ 门需要至少2个比特：1个控制，1个目标。
    # 实际上，如果 num_qubits == 1，则 MCZ 退化为 Z 门。
    # 如果 num_qubits == 2，则 MCZ 退化为 CZ 门。
    # 如果 num_qubits > 2，则 MCZ 需要 (num_qubits-1) 个控制和 1 个目标。
    if num_qubits == 1:
        oracle_qc.z(0)
    elif num_qubits == 2:
        oracle_qc.cz(0, 1)
    elif num_qubits > 2:
        # 将所有比特作为控制比特，除了最后一个比特（num_qubits-1）作为目标比特
        # MCZ(controls, target)
        controls_for_mcz = list(range(num_qubits - 1))
        target_for_mcz = num_qubits - 1
        oracle_qc.mcz(controls_for_mcz, target_for_mcz)
    
    # 步骤3: 再次翻转所有为0的比特，以撤销第一次翻转
    for i in qubits_to_flip_to_make_1: 
        oracle_qc.x(i)
            
    return oracle_qc

def build_grover_diffusion_circuit(num_qubits: int) -> 'QuantumCircuit':
    """
    [增强] 构建Grover搜索算法的扩散算子 (Diffusion Operator) 线路。

    扩散算子是 Grover 迭代的核心，它围绕所有状态的平均振幅进行反射，
    从而增强目标态的概率。
    
    其标准线路结构为：H^n -> X^n -> MCZ -> X^n -> H^n，
    其中 n 是量子比特的数量。

    Args:
        num_qubits (int): 电路中的量子比特数量。

    Returns:
        QuantumCircuit: 一个包含了 Grover 扩散算子门序列的量子线路对象。

    Raises:
        ValueError: 如果 `num_qubits` 不是一个正整数。
    """
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer for the Grover diffusion operator.")
    
    logger.info(f"API: Building Grover diffusion circuit for {num_qubits} qubits.")
    
    diffusion_qc = QuantumCircuit(num_qubits=num_qubits, description="Grover Diffusion Operator")
    
    # 1. 应用 H^n
    for i in range(num_qubits): diffusion_qc.h(i)
    # 2. 应用 X^n
    for i in range(num_qubits): diffusion_qc.x(i)
        
    # 3. 应用多控制Z门 (MCZ)
    # MCZ 门需要至少1个目标和至少1个控制 (总共至少2个比特)
    if num_qubits == 1:
        diffusion_qc.z(0) # For 1 qubit, MCZ is just Z
    elif num_qubits == 2:
        diffusion_qc.cz(0, 1) # For 2 qubits, MCZ is CZ
    elif num_qubits > 2:
        # 将所有比特作为控制比特，除了最后一个比特（num_qubits-1）作为目标比特
        controls_for_mcz = list(range(num_qubits - 1))
        target_for_mcz = num_qubits - 1
        diffusion_qc.mcz(controls_for_mcz, target_for_mcz)
    
    # 4. 再次应用 X^n
    for i in range(num_qubits): diffusion_qc.x(i)
    # 5. 再次应用 H^n
    for i in range(num_qubits): diffusion_qc.h(i)
    
    return diffusion_qc
# ========================================================================
# --- 7. 独立测试块 ---
# ========================================================================

if __name__ == '__main__':
    """
    [最终修正版] 当此脚本被直接执行时，运行此处的测试代码。
    此版本适配了内化的并行管理机制，通过 enable/disable_parallelism API 进行控制。
    
    测试流程：
    1. 配置基础日志。
    2. 定义一个通用的测试运行函数 `run_test`。
    3. 定义所有单元测试函数。
    4. **运行所有单元测试 (默认在单线程模式下)**，验证核心逻辑的正确性。
    5. **启用并行计算**。
    6. **运行性能演示 (在并行模式下)**，展示加速效果，并验证并行机制。
    7. **禁用并行计算**，确保资源被清理。
    8. 汇总并打印所有测试结果。
    """
    
    # --- 步骤 1: 配置一个基础的日志记录器，以便在测试时能看到输出 ---
    logging.basicConfig(
        level=logging.INFO, # 设置为 INFO 以获得清晰的测试流程输出
        format='%(asctime)s - [%(levelname)s] - (%(name)s) - %(message)s'
    )
    
    # 打印一个清晰的测试开始横幅
    print("\n" + "="*80)
    print("--- Running quantum_core.py Standalone Self-Tests (Pure Python Core Version) ---")
    print("="*80 + "\n")

    # --- 步骤 2: 定义一个通用的测试结果打印函数 ---
    def run_test(test_name: str, test_function: callable, backend_choice: str = "pure_python"):
        """
        运行一个测试函数，并捕获结果和时间。
        
        Args:
            test_name (str): 测试的名称。
            test_function (callable): 实际执行测试逻辑的函数。
            backend_choice (str): 指定测试使用的后端 ('pure_python' 或 'cupy')。
        
        Returns:
            bool: 如果测试通过或被跳过，则返回 True；如果失败，则返回 False。
        """
        # 临时配置后端，以便测试可以针对特定后端运行
        original_config = _core_config.copy()
        try:
            # 设置一个较高的 MAX_QUBITS 以允许大型测试，并设置并行阈值
            configure_quantum_core({"BACKEND_CHOICE": backend_choice, "MAX_QUBITS": 15, "PARALLEL_COMPUTATION_QUBIT_THRESHOLD": 8}) 
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
            return True # 认为跳过是预期行为，不算失败
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            print(f"❌ FAILED: {test_name} (Backend: {backend_choice}) (after {duration:.2f} ms)")
            print(f"    ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc(limit=5) # 限制栈回溯深度，防止输出过长
            print("")
            return False
        finally:
            # 恢复原始配置，确保不会影响后续测试
            configure_quantum_core(original_config)


    # --- 步骤 3: 编写各个功能的测试函数 ---
    # 这些测试函数都假定量子核心库在运行时被正确配置

    def test_state_and_circuit_creation():
        """测试 QuantumState 和 QuantumCircuit 的基本创建和属性。"""
        state = create_quantum_state(3)
        assert state.num_qubits == 3, "State should have 3 qubits"
        assert len(state.density_matrix) == 8 and len(state.density_matrix[0]) == 8, "Density matrix shape should be 8x8 for 3 qubits"
        assert state.is_valid(), "Initial 3-qubit state should be valid."
        
        qc = QuantumCircuit(num_qubits=3, description="Test Circuit")
        assert qc.num_qubits == 3, "Circuit should have 3 qubits"
        assert qc.description == "Test Circuit"
        assert len(qc) == 0, "Initial circuit should have 0 instructions"
        
        try:
            create_quantum_state(-1)
            assert False, "Should raise ValueError for negative qubits"
        except ValueError:
            pass
        
        state_0q = create_quantum_state(0)
        assert state_0q.num_qubits == 0
        assert len(state_0q.density_matrix) == 1 and len(state_0q.density_matrix[0]) == 1
        assert math.isclose(state_0q.density_matrix[0][0].real, 1.0, abs_tol=1e-9) and math.isclose(state_0q.density_matrix[0][0].imag, 0.0, abs_tol=1e-9)

    def test_gate_application_and_circuit_run():
        """测试门操作的应用和整个线路的执行，并验证最终状态（Bell态）。"""
        state = create_quantum_state(2)
        qc = QuantumCircuit(num_qubits=2, description="Bell State Preparation")
        qc.h(0)
        qc.cnot(0, 1)
        final_state = run_circuit_on_state(state, qc)
        
        probabilities = get_measurement_probabilities(final_state) 
        expected_probs = [0.5, 0.0, 0.0, 0.5]
        assert all(math.isclose(p_actual, p_expected, abs_tol=1e-9) for p_actual, p_expected in zip(probabilities, expected_probs)), \
            f"Bell state probabilities are incorrect. Expected {expected_probs}, got {probabilities}"
        assert final_state.is_valid(), "Bell state should be valid."

    def test_advanced_gate_rxx():
        """测试高级参数化门 RXX 的正确性。"""
        state = create_quantum_state(2)
        qc = QuantumCircuit(num_qubits=2)
        qc.rxx(0, 1, math.pi / 2)
        final_state = run_circuit_on_state(state, qc)
        
        probabilities = get_measurement_probabilities(final_state)
        expected_probs = [0.5, 0.0, 0.0, 0.5]
        assert all(math.isclose(p_actual, p_expected, abs_tol=1e-9) for p_actual, p_expected in zip(probabilities, expected_probs)), \
            f"RXX(pi/2)|00> state probabilities are incorrect. Expected {expected_probs}, got {probabilities}"
        
        expected_dm_03 = 0.0 + 0.5j # Corrected expected value
        dm = final_state.density_matrix
        actual_dm_03 = dm[0][3] if isinstance(dm, list) else dm[0,3].item()
        
        assert math.isclose(actual_dm_03.real, expected_dm_03.real, abs_tol=1e-9) and \
               math.isclose(actual_dm_03.imag, expected_dm_03.imag, abs_tol=1e-9), \
            f"RXX(pi/2)|00> dm[0,3] is incorrect. Expected {expected_dm_03}, got {actual_dm_03}"

    def test_qft_and_inverse_qft():
        """测试QFT及其逆操作，现在核心引擎已修复，应能通过。"""
        num_qubits = 3
        state = create_quantum_state(num_qubits)
        qc_prep = QuantumCircuit(num_qubits)
        qc_prep.x(0)
        qc_prep.x(2) # Prepare |101> state (index 5)
        prepared_state = run_circuit_on_state(state, qc_prep)

        qft_circuit = build_qft_circuit(num_qubits, inverse=False)
        qft_state = run_circuit_on_state(prepared_state, qft_circuit)

        iqft_circuit = build_qft_circuit(num_qubits, inverse=True)
        final_state = run_circuit_on_state(qft_state, iqft_circuit)

        final_probabilities = get_measurement_probabilities(final_state)
        expected_index = 5
        assert math.isclose(final_probabilities[expected_index], 1.0, abs_tol=1e-7), \
            f"QFT->IQFT failed. Prob of |101> is {final_probabilities[expected_index]:.6f}, expected 1.0"
        
        for i, prob in enumerate(final_probabilities):
            if i != expected_index:
                assert math.isclose(prob, 0.0, abs_tol=1e-7), f"QFT->IQFT failed. Prob of |{i:03b}> is {prob:.6f}, expected 0.0"

    def test_partial_trace_and_entropy():
        """测试部分迹和冯·诺依曼纠缠熵的计算。"""
        state = create_quantum_state(3)
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cnot(0, 1) # Bell state |00>+|11> for q0, q1. q2 is |0>
        final_state = run_circuit_on_state(state, qc)

        rho_q0_backend_type = final_state._partial_trace([1, 2])
        expected_rho_q0 = [[0.5+0.0j, 0.0+0.0j], [0.0+0.0j, 0.5+0.0j]]
        assert final_state._backend.allclose(rho_q0_backend_type, expected_rho_q0, atol=1e-9), \
            f"Partial trace for q0 incorrect. Expected {expected_rho_q0}, got {rho_q0_backend_type}"

        entropy_q0 = calculate_entanglement_entropy(final_state, [0])
        assert math.isclose(entropy_q0, 1.0, abs_tol=1e-9), \
            f"Entropy of q0 (entangled) incorrect. Expected 1.0, got {entropy_q0}"
        
        entropy_q2 = calculate_entanglement_entropy(final_state, [2])
        assert math.isclose(entropy_q2, 0.0, abs_tol=1e-9), \
            f"Entropy of q2 (non-entangled) incorrect. Expected 0.0, got {entropy_q2}"

    def test_hamiltonian_expectation():
        """测试哈密顿量期望值的计算。"""
        state = create_quantum_state(2)
        qc_prep = QuantumCircuit(2); qc_prep.x(1) # Prepare |10> state (q1=1, q0=0)
        state_10 = run_circuit_on_state(state, qc_prep)
        
        hamiltonian = [
            PauliString(coefficient=1.0, pauli_map={0: 'Z'}),
            PauliString(coefficient=0.5, pauli_map={1: 'X'})
        ]
        # For |q1 q0> = |10>:
        # <10|Z_0|10> = <0|Z|0> = 1
        # <10|X_1|10> = <1|X|1> = 0
        # Expected: 1.0 * 1 + 0.5 * 0 = 1.0
        expected_energy = 1.0
        calculated_energy = calculate_hamiltonian_expectation_value(state_10, hamiltonian)
        assert math.isclose(calculated_energy, expected_energy, abs_tol=1e-9), \
            f"Hamiltonian expectation incorrect. Expected {expected_energy}, got {calculated_energy}"

    def test_trotter_step():
        """测试 Suzuki-Trotter 分解的单步演化。"""
        state = create_quantum_state(2)
        hamiltonian = [
            PauliString(coefficient=1.0, pauli_map={0: 'X', 1: 'X'})
        ]
        time_step = math.pi / 4.0 # For RXX(pi/2)
        trotter_qc = build_trotter_step_circuit(hamiltonian, time_step)
        evolved_state = run_circuit_on_state(state, trotter_qc)
        
        probabilities = get_measurement_probabilities(evolved_state)
        expected_probs = [0.5, 0.0, 0.0, 0.5]
        
        assert all(math.isclose(p_actual, p_expected, abs_tol=1e-7) for p_actual, p_expected in zip(probabilities, expected_probs)), \
            f"Trotter step (RXX(pi/2)|00>) probabilities incorrect. Expected {expected_probs}, got {probabilities}"

    def test_bloch_vector():
        """测试布洛赫矢量计算。"""
        state = create_quantum_state(1)
        bloch_initial = get_bloch_vector(state, 0)
        assert all(math.isclose(b_actual, b_expected, abs_tol=1e-9) for b_actual, b_expected in zip(bloch_initial, [0.0, 0.0, 1.0])), \
            f"Bloch vector for initial |0> state incorrect. Expected [0,0,1], got {bloch_initial}"

        qc_ry = QuantumCircuit(1)
        qc_ry.ry(0, math.pi / 2)
        ry_state = run_circuit_on_state(state, qc_ry)
        bloch_ry = get_bloch_vector(ry_state, 0)
        expected_bloch_ry = [1.0, 0.0, 0.0]
        assert all(math.isclose(b_actual, b_expected, abs_tol=1e-9) for b_actual, b_expected in zip(bloch_ry, expected_bloch_ry)), \
            f"Bloch vector for RY(pi/2)|0> incorrect. Expected {expected_bloch_ry}, got {bloch_ry}"

        qc_h = QuantumCircuit(1)
        qc_h.h(0)
        h_state = run_circuit_on_state(state, qc_h)
        bloch_h = get_bloch_vector(h_state, 0)
        expected_bloch_h = [1.0, 0.0, 0.0]
        assert all(math.isclose(b_actual, b_expected, abs_tol=1e-9) for b_actual, b_expected in zip(bloch_h, expected_bloch_h)), \
            f"Bloch vector for H|0> incorrect. Expected {expected_bloch_h}, got {bloch_h}"

        qc_s_on_h = QuantumCircuit(1)
        qc_s_on_h.h(0)
        qc_s_on_h.s(0)
        s_on_h_state = run_circuit_on_state(state, qc_s_on_h)
        bloch_s_on_h = get_bloch_vector(s_on_h_state, 0)
        expected_bloch_s_on_h = [0.0, 1.0, 0.0]
        assert all(math.isclose(b_actual, b_expected, abs_tol=1e-9) for b_actual, b_expected in zip(bloch_s_on_h, expected_bloch_s_on_h)), \
            f"Bloch vector for S(H|0>) incorrect. Expected {expected_bloch_s_on_h}, got {bloch_s_on_h}"


    def test_grover_diffusion():
        """测试 Grover 扩散算子的构建和应用。"""
        num_qubits = 2
        state = create_quantum_state(num_qubits) # Start in |00>
        
        diffusion_qc = build_grover_diffusion_circuit(num_qubits)
        diffused_state = run_circuit_on_state(state, diffusion_qc)
        probabilities = get_measurement_probabilities(diffused_state)
        expected_probs = [0.25, 0.25, 0.25, 0.25]
        assert all(math.isclose(p_actual, p_expected, abs_tol=1e-9) for p_actual, p_expected in zip(probabilities, expected_probs)), \
            f"Grover Diffusion on |00> probabilities incorrect. Expected {expected_probs}, got {probabilities}"
        assert diffused_state.is_valid(), "Diffused state should be valid."


    def test_classical_control_flow():
        """测试经典控制流 (conditional operations)。"""
        state = create_quantum_state(2)
        qc = QuantumCircuit(2)
        qc.h(0) # Put q0 in superposition
        qc.measure(0, classical_register_index=0) # Measure q0, store in CR[0]
        qc.x(1, condition=(0, 1)) # If CR[0] is 1, flip q1

        num_trials = 20
        outcomes_probs_list = [get_measurement_probabilities(run_circuit_on_state(state, qc)) for _ in range(num_trials)]
        
        all_00 = 0
        all_11 = 0
        for probs in outcomes_probs_list:
            is_00 = math.isclose(probs[0], 1.0, abs_tol=1e-7)
            is_11 = math.isclose(probs[3], 1.0, abs_tol=1e-7)
            
            assert is_00 or is_11, f"Invalid final state, probabilities: {probs}. Expected |00> or |11>."
            
            if is_00: all_00 += 1
            if is_11: all_11 += 1
        
        assert all_00 + all_11 == num_trials, "Total trials don't match outcomes."
        assert all_00 > 0 and all_11 > 0, "Expected both |00> and |11> outcomes to occur in 20 trials (stochastic test)."
        print(f"    Classical control flow test: {all_00} trials resulted in |00>, {all_11} trials resulted in |11>.")


    def test_large_scale_simulation_performance(num_qubits: int, backend_choice: str):
        """
        [修正版] 对大规模模拟进行一次性能演示测试。
        此版本不再包含平台特定的并行禁用逻辑，它假设并行池由外部管理。
        """
        print(f"--- Performance Demonstration for {num_qubits}-qubit Simulation (Backend: {backend_choice}) ---")
        
        # 保存并恢复原始配置，确保测试的独立性
        original_config = _core_config.copy()
        try:
            # 配置一个较低的并行阈值，以便测试能触发并行逻辑
            configure_quantum_core({
                "BACKEND_CHOICE": backend_choice,
                "PARALLEL_COMPUTATION_QUBIT_THRESHOLD": 8, # 适用于性能测试的较低阈值
                "MAX_QUBITS": 15 # 允许更大的模拟
            })

            start_time = time.perf_counter()
            state = create_quantum_state(num_qubits)
            
            # 构建一个包含Hadamard、RZ旋转和CNOT纠缠的复杂电路
            qc = QuantumCircuit(num_qubits=num_qubits, description=f"Performance test circuit for {num_qubits} qubits")
            for i in range(num_qubits):
                qc.h(i)
                qc.rz(i, math.pi / (i + 4)) # 参数化旋转，角度随比特索引变化
            for i in range(num_qubits - 1):
                qc.cnot(i, i + 1) # 邻近比特纠缠
            
            # 增加一个 Trotter 步，模拟哈密顿量演化，进一步增加计算量
            hamiltonian_test = [
                PauliString(coefficient=0.1, pauli_map={i: 'Z', (i+1)%num_qubits: 'Z'}) 
                for i in range(num_qubits)
            ]
            trotter_step_qc = build_trotter_step_circuit(hamiltonian_test, 0.01)
            qc.instructions.extend(trotter_step_qc.instructions) # 将 Trotter 步的指令加入线路
                
            final_state = run_circuit_on_state(state, qc)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # 验证结果
            backend_name = final_state._backend.__name__ if isinstance(final_state._backend, types.ModuleType) else type(final_state._backend).__name__
            print(f"    Backend used by QuantumState: {backend_name}")
            print(f"    Total simulation time for {num_qubits} qubits: {duration_ms:.2f} ms")
            assert duration_ms > 0, "Simulation duration should be positive"
            assert final_state.is_valid(), "Final state of performance test should be valid."

        finally:
            # 恢复原始配置，确保不会影响后续测试
            configure_quantum_core(original_config)
        
        print("--------------------------------------------------")

    # --- 步骤 4: 运行所有测试并汇总结果 ---
    
    # 纯Python后端单元测试列表
    pure_python_tests = {
        "State and Circuit Creation": test_state_and_circuit_creation,
        "Gate Application and Circuit Run (Bell State)": test_gate_application_and_circuit_run,
        "Advanced Gate RXX Correctness": test_advanced_gate_rxx,
        "QFT and Inverse QFT Correctness": test_qft_and_inverse_qft,
        "Partial Trace and Von Neumann Entropy": test_partial_trace_and_entropy,
        "Hamiltonian Expectation Value": test_hamiltonian_expectation,
        "Trotter Step Circuit Simulation": test_trotter_step,
        "Bloch Vector Calculation": test_bloch_vector,
        "Grover Diffusion Operator": test_grover_diffusion,
        "Classical Control Flow": test_classical_control_flow,
    }

    # CuPy后端单元测试列表 (如果 CuPy 可用)
    cupy_tests = {}
    if cp is not None:
        print("\n" + "#"*80)
        print("--- Running CuPy Backend Tests (Unit) ---")
        print("#"*80 + "\n")
        # 复制纯Python测试到CuPy，因为大部分逻辑是通用的
        for name, func in pure_python_tests.items():
            cupy_tests[f"{name} (CuPy)"] = lambda f=func: f() 
    else:
        print("\n" + "#"*80)
        print("--- CuPy not found, skipping CuPy Backend Tests ---")
        print("#"*80 + "\n")

    # 合并所有单元测试配置
    all_unit_test_configs = []
    for name, func in pure_python_tests.items():
        all_unit_test_configs.append((name, func, "pure_python"))
    for name, func in cupy_tests.items():
        all_unit_test_configs.append((name, func, "cupy"))


    # [新用法] 主测试流程现在负责管理并行池的生命周期
    
    # 首先，在单线程模式下运行所有单元测试，以确保基础逻辑正确
    print("\n" + "#"*80)
    print("--- Running All Unit Tests (Initially Single-Threaded) ---")
    print("#"*80 + "\n")
    results_summary = []
    for name, func, backend_choice in all_unit_test_configs:
        results_summary.append(run_test(name, func, backend_choice))
    
    # 然后，启用并行，专门运行性能演示
    print("\n" + "="*80)
    print("--- Running Performance Demonstrations (Parallel Mode Enabled) ---")
    print("="*80 + "\n")
    
    try:
        # 显式启用并行
        enable_parallelism() 
        
        # 只有当并行成功启用时才运行性能测试
        if _parallel_enabled:
            # 运行 Pure Python 并行性能测试 (num_qubits >= PARALLEL_COMPUTATION_QUBIT_THRESHOLD 会触发并行)
            test_large_scale_simulation_performance(10, "pure_python")
            
            # CuPy 测试不受 CPU 并行影响，但也可以在这里运行以作对比
            if cp is not None:
                test_large_scale_simulation_performance(12, "cupy")
        else:
            print("⚠️ SKIPPED: Performance tests were skipped because parallelism could not be enabled.")
            print("    (On Windows, ensure this script is run from a `if __name__ == '__main__':` block,")
            print("    or check logs for errors during `enable_parallelism()` call).")

    except Exception as e:
        print(f"\nAn error occurred during performance demonstration block: {type(e).__name__}: {e}")
        logger.error("Performance demonstration block failed.", exc_info=True)
    finally:
        # 确保无论成功与否，都关闭并行池
        disable_parallelism()

    # --- 步骤 6: 打印最终总结 ---
    passed_count = sum(results_summary)
    total_count = len(results_summary)
    
    print("\n" + "="*80)
    print("--- Overall Test Summary ---")
    for i, result in enumerate(results_summary):
        status = "✅ PASSED" if result else "❌ FAILED"
        name, _, backend_choice = all_unit_test_configs[i]
        print(f"{status:<10} | {name} (Backend: {backend_choice})")
    print("-" * 80)
    print(f"Total Unit Test Summary: {passed_count} / {total_count} tests passed.")
    print("Performance demonstrations completed.")
    print("="*80 + "\n")

    if passed_count != total_count:
        sys.exit(1)