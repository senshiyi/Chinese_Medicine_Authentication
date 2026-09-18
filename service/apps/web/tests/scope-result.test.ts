// @vitest-environment jsdom
import { expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import CaseResultView from "../src/views/CaseResultView.vue";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../src/api", () => ({
  ApiError: class extends Error {},
  api: {
    getCase: vi.fn().mockResolvedValue({
      case: { status: "completed", result_type: "authentic_likely" },
      result_verdict_valid: false,
      scope_enforced: true,
      scope: { scope: "unavailable" },
      prediction: { authenticity_label: "authentic", confidence: 0.999 },
      prediction_message: "此药品为正品，有99.9%的可能该药材为正品。",
      images: [], review_requests: [], expert_reviews: [],
      consent: { research_training: false }, failure: null,
    }),
  },
}));

it("hides an affirmative historical message even if the API returns it", async () => {
  const wrapper = mount(CaseResultView, { props: { caseId: "historical" } });
  await flushPromises();
  expect(wrapper.text()).toContain("无法判断");
  expect(wrapper.text()).not.toContain("此药品为正品");
  expect(wrapper.text()).not.toContain("99.9%");
  wrapper.unmount();
});
