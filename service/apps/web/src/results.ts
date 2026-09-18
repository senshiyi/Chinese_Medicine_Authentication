// 实验范围与结果文案（与后端预注册固定文案保持一致；不在前端拼接模型结论）
export const EXPERIMENTAL_NOTICE =
  "本服务为阶段A实验性闭集服务，仅支持实验室白底拍摄范围；不识别伪品实际物种。模型结果不能替代专业鉴定。";

export const RESULT_DISPLAY: Record<string, string> = {
  authentic_likely: "正品特征吻合度高",
  suspected_counterfeit: "疑似伪品（无法确认实际物种，请人工复核）",
  inconclusive: "无法判断",
};

export const STATUS_TEXT: Record<string, string> = {
  draft: "草稿",
  ready: "可分析",
  queued: "排队中",
  analyzing: "分析中",
  completed: "已完成",
  needs_retake: "需要补拍",
  inconclusive: "无法判断",
  failed: "失败",
};

export interface OutcomeInfo {
  display: string;
  tone: "positive" | "warning" | "neutral" | "error";
  scope?: string | null;
  notice?: string | null;
}

export function outcomeInfo(resultType: string | null, failed: boolean): OutcomeInfo {
  if (failed) return { display: "分析失败（工程错误，无真伪结论）", tone: "error" };
  switch (resultType) {
    case "authentic_likely":
      return { display: RESULT_DISPLAY.authentic_likely, tone: "positive" };
    case "suspected_counterfeit":
      return { display: RESULT_DISPLAY.suspected_counterfeit, tone: "warning" };
    case "inconclusive":
      return { display: RESULT_DISPLAY.inconclusive, tone: "neutral" };
    default:
      return { display: "等待分析", tone: "neutral" };
  }
}

// 范围门控文案：无范围证据时不呈现肯定性真伪结论。
export const SCOPE_NOTICE_UNAVAILABLE =
  "当前无法提供可验证的范围证据，不给出真伪结论（不能据此判定为本药材，也不能据此排除）。";
export const SCOPE_NOTICE_REJECTED =
  "该图片被判定为超出当前范围门控，不给出真伪结论。";

export function verdictInfo(
  resultType: string | null,
  failed: boolean,
  scopeEnforced: boolean,
  verdictValid: boolean,
  scope: string | null | undefined
): OutcomeInfo {
  const base = outcomeInfo(resultType, failed);
  if (failed) return base;
  if (scopeEnforced && !verdictValid) {
    const notice =
      scope === "rejected" ? SCOPE_NOTICE_REJECTED : SCOPE_NOTICE_UNAVAILABLE;
    return { ...base, display: RESULT_DISPLAY.inconclusive, tone: "neutral", scope, notice };
  }
  return { ...base, scope };
}
