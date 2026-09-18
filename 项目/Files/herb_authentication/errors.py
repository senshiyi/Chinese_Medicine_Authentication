"""项目领域异常。

CLI 与未来 Web 适配层只需要捕获这些异常，不应依赖底层库的异常文本。
"""


class HerbAuthenticationError(Exception):
    """所有可预期项目错误的基类。"""


class ConfigurationError(HerbAuthenticationError):
    """配置缺失、类型错误或值超出允许范围。"""


class ManifestValidationError(HerbAuthenticationError):
    """数据清单未满足训练或评估契约。"""


class EnvironmentCompatibilityError(HerbAuthenticationError):
    """Python、依赖或设备环境不兼容。"""


class InputImageError(HerbAuthenticationError):
    """输入图片损坏、超限或不可解码。"""


class CheckpointCompatibilityError(HerbAuthenticationError):
    """Checkpoint schema、类别或预处理契约不兼容。"""
