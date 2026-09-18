// CaseView 回归：首次加载失败必须终止 loading 并进入错误态（不再永远“加载中”）。
// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import CaseView from "../src/views/CaseView.vue";

const mockGetCase = vi.fn();

vi.mock("../src/api", () => ({
  api: { getCase: (...args: unknown[]) => mockGetCase(...args) },
  ApiError: class ApiError extends Error {
    status: number;
    code: string;
    constructor(status: number, code: string, message: string) {
      super(message);
      this.status = status;
      this.code = code;
    }
  },
}));

describe("CaseView load lifecycle", () => {
  it("terminates loading and shows error state when initial load fails", async () => {
    mockGetCase.mockRejectedValue(new Error("network down"));
    const wrapper = mount(CaseView, { props: { id: "missing-case" } });

    expect(wrapper.text()).toContain("加载中");

    await flushPromises();
    expect(wrapper.text()).not.toContain("加载中");
    expect(wrapper.text()).toContain("加载失败或无权访问");
    expect(wrapper.find(".error").exists()).toBe(true);
  });
});
