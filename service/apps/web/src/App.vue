<script setup lang="ts">
import { ref, onMounted } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api, ApiError, type User } from "./api";

// 简单的会话响应式状态（真正鉴权在路由守卫中完成）
const user = ref<User | null>(null);
const route = useRoute();
const router = useRouter();

async function loadUser(): Promise<void> {
  try {
    const me = await api.me();
    user.value = me.user;
  } catch {
    user.value = null;
  }
}

async function logout(): Promise<void> {
  try {
    await api.logout();
  } catch (error) {
    if (!(error instanceof ApiError && error.status === 401)) {
      console.warn("logout:", error);
    }
  }
  user.value = null;
  router.push({ name: "login" });
}

onMounted(loadUser);
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <div class="brand">
        <strong>骨碎补鉴伪</strong>
        <span class="badge">阶段A实验</span>
      </div>
      <nav v-if="user" class="nav">
        <RouterLink :to="{ name: 'cases' }">我的案件</RouterLink>
        <RouterLink
          v-if="user.role === 'expert' || user.role === 'admin'"
          :to="{ name: 'reviews' }"
        >
          专家复核
        </RouterLink>
        <button class="link-btn" type="button" @click="logout">退出</button>
      </nav>
    </header>

    <RouterView />

    <footer v-if="user" class="app-footer">
      <p>
        实验性服务 · 闭集范围 · 不识别伪品实际物种 · 结果不能替代专业鉴定（药典/显微/理化/分子检测）
      </p>
    </footer>
  </div>
</template>
