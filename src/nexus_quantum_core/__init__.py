# src/nexus_quantum_core/__init__.py

# 从 quantum_core.py 模块中导入你希望公开的类和函数
from .quantum_core import (
    QuantumCircuit,
    QuantumState,
    create_quantum_state,
    run_circuit_on_state,
    get_measurement_probabilities,
    get_marginal_probability,
    set_process_pool
)

# (可选) 定义 __all__ 来明确指定哪些名称会被 from nexus_quantum_core import * 导入
__all__ = [
    "QuantumCircuit",
    "QuantumState",
    "create_quantum_state",
    "run_circuit_on_state",
    "get_measurement_probabilities",
    "get_marginal_probability",
    "set_process_pool",
]