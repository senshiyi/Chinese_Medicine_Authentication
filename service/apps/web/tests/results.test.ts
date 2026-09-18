import { describe, expect, it } from "vitest";
import {
  EXPERIMENTAL_NOTICE,
  RESULT_DISPLAY,
  outcomeInfo,
  verdictInfo,
  STATUS_TEXT,
  SCOPE_NOTICE_UNAVAILABLE,
  SCOPE_NOTICE_REJECTED,
} from "../src/results";

describe("result display copy", () => {
  it("provides approved closed-set copy without claiming species", () => {
    expect(EXPERIMENTAL_NOTICE).toContain("阶段A");
    expect(EXPERIMENTAL_NOTICE).toContain("不识别伪品实际物种");
    expect(RESULT_DISPLAY.authentic_likely).toBe("正品特征吻合度高");
    expect(RESULT_DISPLAY.suspected_counterfeit).toContain("疑似伪品");
    expect(RESULT_DISPLAY.inconclusive).toBe("无法判断");
  });

  it("maps outcome type to tone and display", () => {
    expect(outcomeInfo("authentic_likely", false)).toEqual({
      display: "正品特征吻合度高",
      tone: "positive",
    });
    expect(outcomeInfo("suspected_counterfeit", false).tone).toBe("warning");
    expect(outcomeInfo("inconclusive", false).display).toBe("无法判断");
    expect(outcomeInfo(null, false).display).toBe("等待分析");
  });

  it("marks failed as engineering error with no conclusion", () => {
    const info = outcomeInfo(null, true);
    expect(info.tone).toBe("error");
    expect(info.display).not.toContain("正品");
    expect(info.display).not.toContain("伪品");
    expect(info.display).toContain("工程错误");
  });

  it("hides affirmative verdict when scope enforced and evidence absent", () => {
    const info = verdictInfo("authentic_likely", false, true, false, "unavailable");
    expect(info.display).toBe("无法判断");
    expect(info.tone).toBe("neutral");
    expect(info.notice).toBeTruthy();
    expect(info.scope).toBe("unavailable");
  });

  it("shows rejected scope notice when out of scope", () => {
    const info = verdictInfo("authentic_likely", false, true, false, "rejected");
    expect(info.display).toBe("无法判断");
    expect(info.notice).toBe(SCOPE_NOTICE_REJECTED);
  });

  it("keeps affirmative verdict when scope accepted or not enforced", () => {
    expect(verdictInfo("authentic_likely", false, true, true, "accepted").display).toBe(
      "正品特征吻合度高"
    );
    expect(verdictInfo("authentic_likely", false, false, true, null).display).toBe(
      "正品特征吻合度高"
    );
    expect(verdictInfo("authentic_likely", false, false, true, null).notice).toBeFalsy();
    expect(SCOPE_NOTICE_UNAVAILABLE).toContain("范围");
  });

  it("covers all status labels", () => {
    for (const key of [
      "draft",
      "ready",
      "queued",
      "analyzing",
      "completed",
      "needs_retake",
      "inconclusive",
      "failed",
    ]) {
      expect(STATUS_TEXT[key]).toBeTruthy();
    }
  });
});
