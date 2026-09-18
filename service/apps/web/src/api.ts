// API 客户端：Cookie 会话 + CSRF 头 + 版本化 /api/v1 接口。
const BASE = "/api/v1";

let csrfToken: string | null = null;
// 会话（含 CSRF）更新的唯一入口由 /auth/me 返回；me 请求单飞（single-flight）：
// 路由守卫 bootSession 与 App 挂载同时触发时只发一个请求，避免两次 /me 竞态。
let meRequest: Promise<MeResponse> | null = null;

export interface MeResponse {
  user: User;
  csrf_token: string;
}

export function setCsrfToken(token: string | null): void {
  csrfToken = token;
}

export function getCsrfToken(): string | null {
  return csrfToken;
}

/** 任何登录/注册/me 响应都统一经此更新客户端 token（集中式会话更新）。 */
function adoptSession(data: MeResponse): void {
  if (data.csrf_token) {
    setCsrfToken(data.csrf_token);
  }
}

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

interface RequestOptions {
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
  form?: FormData;
  csrf?: boolean;
}

export async function request<T = unknown>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const { method = "GET", body, form, csrf = true } = options;
  const headers: Record<string, string> = {};
  if (body !== undefined && !form) {
    headers["Content-Type"] = "application/json";
  }
  if (csrf && method !== "GET" && csrfToken) {
    headers["X-CSRF-Token"] = csrfToken;
  }
  const payload =
    form !== undefined
      ? form
      : body !== undefined
        ? JSON.stringify(body)
        : undefined;
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers,
    credentials: "include",
    body: payload,
  });
  if (response.status === 204) {
    return undefined as T;
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = (data as { detail?: { code?: string; message?: string } }).detail;
    throw new ApiError(
      response.status,
      detail?.code || "http_error",
      detail?.message || `请求失败（${response.status}）`
    );
  }
  return data as T;
}

export function refreshCsrf(): Promise<string | null> {
  return api
    .me()
    .then((data) => data.csrf_token || null)
    .catch(() => null);
}

// 类型
export interface User {
  id: string;
  email: string;
  role: "user" | "expert" | "admin";
  display_name: string;
}

export interface CaseSummary {
  id: string;
  status: string;
  result_type: string | null;
  claimed_herb: string;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  whole_image_count: number;
  detail_image_count: number;
  cross_section_image_count: number;
  scope_decision: Record<string, unknown> | null;
  scope_enforced: boolean;
  result_verdict_valid: boolean;
}

export interface CaseImage {
  id: string;
  case_id: string;
  view_type: "whole" | "detail" | "cross_section";
  sha256: string;
  width: number;
  height: number;
  format: string;
  size_bytes: number;
  quality: Record<string, unknown> | null;
  quality_flags: string[];
  participates_in_inference: boolean;
  created_at: string;
  content_url: string;
}

export interface Prediction {
  id: string;
  analysis_job_id: string;
  image_id: string;
  view_type: string;
  model_version: string;
  strategy_version: string;
  experimental_closed_set: boolean;
  open_set_calibrated: boolean;
  authenticity_label: "authentic" | "counterfeit";
  confidence: number;
  probabilities: { authentic: number; counterfeit: number };
  actual_species_verified: boolean;
  message: string;
  scope_warning: string | null;
  scope_decision: Record<string, unknown> | null;
  scope_enforced: boolean;
  model_fingerprint: string | null;
  created_at: string;
}

export interface CaseDetail {
  case: CaseSummary;
  images: CaseImage[];
  prediction: Prediction | null;
  prediction_message: string | null;
  prediction_scope_warning: string | null;
  result_display: string | null;
  result_verdict_valid: boolean;
  scope: Record<string, unknown> | null;
  scope_enforced: boolean;
  experimental_notice: string;
  disclaimer: string;
  failure: { category: string; occurred_at: string | null } | null;
  analysis_job_id: string | null;
  analysis_status: string | null;
  review_requests: Array<Record<string, unknown>>;
  expert_reviews: Array<Record<string, unknown>>;
  consent: { analysis: boolean; research_training: boolean };
  deletion_requested: boolean;
}

export interface AnalyzeResult {
  case_id: string;
  analysis_job_id: string;
  job_status: string;
  case_status: string;
  replayed: boolean;
}

export const api = {
  register: async (body: { email: string; password: string; display_name: string }) => {
    const data = await request<MeResponse>("/auth/register", {
      method: "POST",
      body,
      csrf: false,
    });
    adoptSession(data);
    return data;
  },
  login: async (body: { email: string; password: string }) => {
    const data = await request<MeResponse>("/auth/login", {
      method: "POST",
      body,
      csrf: false,
    });
    adoptSession(data);
    return data;
  },
  logout: () => request("/auth/logout", { method: "POST" }),
  me: (): Promise<MeResponse> => {
    if (!meRequest) {
      meRequest = request<MeResponse>("/auth/me", { csrf: false })
        .then((data) => {
          adoptSession(data);
          return data;
        })
        .finally(() => {
          meRequest = null;
        });
    }
    return meRequest;
  },

  listCases: () => request<CaseSummary[]>("/cases"),
  createCase: (researchConsent: boolean) =>
    request<CaseSummary>("/cases", {
      method: "POST",
      body: { claimed_herb: "骨碎补", analysis_consent: true, research_consent: researchConsent },
    }),
  getCase: (caseId: string) => request<CaseDetail>(`/cases/${caseId}`),
  uploadImage: (caseId: string, viewType: string, file: File) => {
    const form = new FormData();
    form.append("view_type", viewType);
    form.append("file", file);
    return request<{ image: CaseImage; case: CaseSummary }>(
      `/cases/${caseId}/images`,
      { method: "POST", form }
    );
  },
  deleteImage: (caseId: string, imageId: string) =>
    request(`/cases/${caseId}/images/${imageId}`, { method: "DELETE" }),
  analyze: (caseId: string) =>
    request<AnalyzeResult>(`/cases/${caseId}/analyze`, { method: "POST" }),
  requestReview: (caseId: string, reason: string) =>
    request(`/cases/${caseId}/review-requests`, { method: "POST", body: { reason } }),
  updateConsent: (caseId: string, granted: boolean) =>
    request(`/cases/${caseId}/consent`, {
      method: "POST",
      body: { consent_type: "research_training", granted },
    }),
  requestDeletion: (caseId: string) =>
    request(`/cases/${caseId}/deletion-request`, { method: "POST" }),

  adminListReviews: (status?: string) =>
    request<Array<Record<string, unknown>>>(
      `/admin/reviews${status ? `?status=${encodeURIComponent(status)}` : ""}`
    ),
  adminReviewDetail: (reviewId: string) =>
    request<Record<string, unknown>>(`/admin/reviews/${reviewId}`),
  adminDecision: (reviewId: string, body: Record<string, unknown>) =>
    request(`/admin/reviews/${reviewId}/decisions`, { method: "POST", body }),
  adminModelVersions: () => request<Array<Record<string, unknown>>>("/admin/model-versions"),
};
