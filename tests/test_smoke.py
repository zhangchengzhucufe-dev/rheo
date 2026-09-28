"""冒烟测试：包骨架可导入、每个占位包有 docstring 说明用途。"""

import importlib

PACKAGES = [
    "kernels",
    "runtime",
    "orchestration",
    "adapters",
    "adapters.verl",
    "adapters.sglang",
    "sim",
    "protocol",
    "bench",
    "bench.analysis",
    "rheotrace",
]


def test_packages_importable():
    for name in PACKAGES:
        mod = importlib.import_module(name)
        assert mod.__doc__, f"{name} 缺少占位 docstring"


def test_no_cross_package_imports_yet():
    """骨架阶段包之间应零依赖：任何一个包 import 失败都不应连带其他包。

    当前只检查可独立导入；后续实现阶段若引入包内依赖，此测试应随之演进而非删除。
    """
    for name in PACKAGES:
        importlib.import_module(name)
