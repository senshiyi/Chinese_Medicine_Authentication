import { createRouter, createWebHistory } from "vue-router";
import { api } from "./api";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/cases" },
    { path: "/login", name: "login", component: () => import("./views/LoginView.vue") },
    { path: "/cases", name: "cases", component: () => import("./views/CaseListView.vue") },
    {
      path: "/cases/:id",
      name: "case",
      component: () => import("./views/CaseView.vue"),
      props: true,
    },
    {
      path: "/admin/reviews",
      name: "reviews",
      component: () => import("./views/AdminReviewView.vue"),
      meta: { roles: ["expert", "admin"] },
    },
  ],
});

export interface SessionUser {
  id: string;
  email: string;
  role: "user" | "expert" | "admin";
  display_name: string;
}

export async function bootSession(): Promise<SessionUser | null> {
  // api.me() 内部统一把响应里的 csrf_token 更新到客户端（集中式会话更新），
  // 且 me 请求单飞：App 挂载与路由守卫的重复调用只发一次请求、不会失配。
  try {
    const data = await api.me();
    return data.user;
  } catch {
    return null;
  }
}

router.beforeEach(async (to) => {
  if (to.path === "/login") return true;
  const user = await bootSession();
  if (!user) return { name: "login", query: { redirect: to.fullPath } };
  const allowed = to.meta.roles as string[] | undefined;
  if (allowed && !allowed.includes(user.role)) return { name: "cases" };
  return true;
});

export default router;
