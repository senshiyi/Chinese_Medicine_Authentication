"""集中生成用户确认过的中文业务结论。"""

from __future__ import annotations

from .results import PredictionResult


class ResultFormatter:
    """将结构化预测转换为精确、稳定且 fail-closed 的中文文案。"""

    def format(self, result: PredictionResult) -> str:
        if not result.is_calibrated or "model_uncalibrated" in result.rejection_reasons:
            return "模型尚未完成真实数据校准，请人工复核。"
        if "actual_species_unknown" in result.rejection_reasons or not result.actual_species:
            return "无法判断该药品的实际药材及真伪，请人工复核。"
        if result.rejected:
            return "无法确定此药品的种类或真伪，请人工复核。"
        if result.authenticity.label == "authentic":
            return f"此药品为正品（置信度：{result.authenticity.confidence:.1%}）。"
        if result.authenticity.label == "counterfeit":
            species = "、".join(item.label for item in result.actual_species)
            return (
                f"此药品为伪品（置信度：{result.authenticity.confidence:.1%}），"
                f"声称为‘{result.claimed_herb.label}’，实际应为‘{species}’。"
            )
        raise ValueError("authenticity label 只允许 authentic 或 counterfeit")
