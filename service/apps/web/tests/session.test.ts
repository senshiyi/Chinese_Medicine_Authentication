// 会话/CSRF 集中化回归：任何 /me（或登录）响应都必须统一更新客户端 token，
// 且并发 /me 只发一个请求——保证“刷新后的 token 用于下一次状态变更”。
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, getCsrfToken, setCsrfToken } from "../src/api";

const USER = { id: "u1", email: "a@example.com", role: "user", display_name: "测试" };

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

const fetchMock = vi.fn(
  async (_input: RequestInfo | URL, _init?: RequestInit): Promise<Response> => {
    const url = String(_input);
    if (url.endsWith("/auth/login")) {
      return jsonResponse({ user: USER, csrf_token: "LOGIN_T1" });
    }
    if (url.endsWith("/auth/me")) {
      return jsonResponse({ user: USER, csrf_token: "ME_T2" });
    }
    if (url.endsWith("/cases") && (_init?.method ?? "GET") === "POST") {
      return jsonResponse({ id: "c1", status: "draft" });
    }
    return jsonResponse({});
  }
);

afterEach(() => {
  vi.unstubAllGlobals();
  setCsrfToken(null);
  fetchMock.mockClear();
});

describe("session token updates are centralized", () => {
  it("adopts login csrf and uses refreshed me token on next mutation", async () => {
    vi.stubGlobal("fetch", fetchMock);

    await api.login({ email: "a@example.com", password: "password123" });
    expect(getCsrfToken()).toBe("LOGIN_T1");

    // 模拟路由守卫 / App 挂载再调一次 /me：响应 token 必须覆盖客户端旧值
    const me = await api.me();
    expect(me.user.email).toBe("a@example.com");
    expect(getCsrfToken()).toBe("ME_T2");

    // 下一次状态变更必须携带刷新后的 token
    await api.createCase(false);
    const mutation = fetchMock.mock.calls.find(
      ([url, init]) =>
        String(url).endsWith("/cases") && (init?.method ?? "GET") === "POST"
    );
    expect(mutation).toBeTruthy();
    const headers = mutation?.[1]?.headers as Record<string, string> | undefined;
    expect(headers?.["X-CSRF-Token"]).toBe("ME_T2");
  });

  it("coalesces overlapping me calls into a single request", async () => {
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockClear();

    const [a, b] = await Promise.all([api.me(), api.me()]);
    expect(a.user.id).toBe(USER.id);
    expect(b.user.id).toBe(USER.id);

    const meCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/auth/me")
    );
    expect(meCalls).toHaveLength(1); // 单飞：App 挂载与 bootSession 只发一个 /me
    expect(getCsrfToken()).toBe("ME_T2");
  });
});
