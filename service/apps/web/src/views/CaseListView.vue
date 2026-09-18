<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { api, ApiError, type CaseSummary } from "../api";
import { STATUS_TEXT } from "../results";
const router = useRouter();
const cases = ref<CaseSummary[]>([]);
const creating = ref(false);
const researchConsent = ref(false);
const errorMsg = ref("");
const loading = ref(true);

async function load(): Promise<void> {
  loading.value = true;
  try {
    cases.value = await api.listCases();
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "加载失败";
  } finally {
    loading.value = false;
  }
}

async function create(): Promise<void> {
  creating.value = true;
  errorMsg.value = "";
  try {
    const created = await api.createCase(researchConsent.value);
    router.push({ name: "case", params: { id: created.id } });
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "创建失败";
  } finally {
    creating.value = false;
  }
}

function tone(status: string, result: string | null): string {
  if (status === "completed" && result === "authentic_likely") return "positive";
  if (status === "completed" && result === "inconclusive") return "";
  if (status === "failed") return "error";
  return "";
}

function resultText(item: CaseSummary): string {
  if (item.scope_enforced && !item.result_verdict_valid) {
    return "无法判断（范围证据缺失或超出范围）";
  }
  if (item.result_type === "authentic_likely") return "正品特征吻合度高";
  if (item.result_type === "inconclusive") return "无法判断";
  return "疑似伪品/其他";
}

onMounted(load);
</script>

<template>
  <section>
    <div class="row-between">
      <h1>我的案件</h1>
      <button class="btn btn-secondary" style="width: auto" type="button" @click="load">刷新</button>
    </div>

    <div class="card">
      <p class="muted">创建新案件，随后按引导上传整根主图（可选局部/断面图）。</p>
      <label class="row-between">
        <span>同意图片分析授权（必须）</span>
        <input type="checkbox" checked disabled />
      </label>
      <label class="row-between">
        <span>同意用于研究训练（可随时撤销）</span>
        <input v-model="researchConsent" type="checkbox" />
      </label>
      <p v-if="errorMsg" class="error">{{ errorMsg }}</p>
      <button class="btn" type="button" :disabled="creating" @click="create">
        {{ creating ? "创建中…" : "新建案件" }}
      </button>
    </div>

    <div v-if="loading" class="muted center">加载中…</div>
    <div v-else-if="cases.length === 0" class="card muted">还没有案件，先新建一个吧。</div>
    <RouterLink
      v-for="item in cases"
      :key="item.id"
      :to="{ name: 'case', params: { id: item.id } }"
      class="card case-row"
    >
      <div class="row-between">
        <strong>#{{ item.id.slice(0, 8) }}</strong>
        <span class="chip" :class="`tone-${tone(item.status, item.result_type)}`">
          {{ STATUS_TEXT[item.status] || item.status }}
        </span>
      </div>
      <div class="muted small">
        主图 {{ item.whole_image_count }} · 局部 {{ item.detail_image_count }} · 断面
        {{ item.cross_section_image_count }}
      </div>
      <div v-if="item.result_type || (item.scope_enforced && !item.result_verdict_valid)" class="small">
        {{ resultText(item) }}
      </div>
    </RouterLink>
  </section>
</template>

<style scoped>
.case-row {
  display: block;
  text-decoration: none;
  color: inherit;
}
</style>
