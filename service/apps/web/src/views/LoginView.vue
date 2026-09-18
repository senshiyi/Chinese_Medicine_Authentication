<script setup lang="ts">
import { ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api, ApiError } from "../api";
import { EXPERIMENTAL_NOTICE } from "../results";

const router = useRouter();
const route = useRoute();
const mode = ref<"login" | "register">("login");
const email = ref("");
const password = ref("");
const displayName = ref("");
const errorMsg = ref("");
const busy = ref(false);

async function submit(): Promise<void> {
  errorMsg.value = "";
  busy.value = true;
  try {
    if (mode.value === "login") {
      await api.login({ email: email.value, password: password.value });
    } else {
      await api.register({
        email: email.value,
        password: password.value,
        display_name: displayName.value,
      });
    }
    // api.login/register 已在内部把会话 CSRF 统一写入客户端
    const target = (route.query.redirect as string) || "/cases";
    router.push(target);
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "网络错误，请稍后再试";
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="card">
    <h1>{{ mode === "login" ? "登录" : "注册" }}</h1>
    <p class="muted">骨碎补（槲蕨）整根外观图像的真伪鉴定的实验性在线流程。</p>

    <div class="row-between">
      <button
        class="btn btn-secondary"
        type="button"
        :disabled="busy"
        style="width: 48%"
        @click="mode = 'login'"
      >
        登录
      </button>
      <button
        class="btn btn-secondary"
        type="button"
        :disabled="busy"
        style="width: 48%"
        @click="mode = 'register'"
      >
        注册
      </button>
    </div>

    <form @submit.prevent="submit">
      <label>邮箱</label>
      <input v-model="email" type="email" required autocomplete="email" />
      <template v-if="mode === 'register'">
        <label>显示名称（可选）</label>
        <input v-model="displayName" type="text" maxlength="120" />
      </template>
      <label>密码{{ mode === "register" ? "（至少 8 位）" : "" }}</label>
      <input v-model="password" type="password" required :minlength="mode === 'register' ? 8 : 1" />
      <p v-if="errorMsg" class="error">{{ errorMsg }}</p>
      <button class="btn" type="submit" :disabled="busy">
        {{ busy ? "处理中…" : mode === "login" ? "登录" : "注册并登录" }}
      </button>
    </form>

    <div class="notice">{{ EXPERIMENTAL_NOTICE }}</div>
  </div>
</template>
