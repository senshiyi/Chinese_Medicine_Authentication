<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { api, type CaseDetail } from "../api";
import CaseCaptureView from "./CaseCaptureView.vue";
import CaseResultView from "./CaseResultView.vue";

const props = defineProps<{ id: string }>();
const detail = ref<CaseDetail | null>(null);
const loading = ref(true);
const loadError = ref("");
let timer: ReturnType<typeof setInterval> | null = null;

const status = computed(() => detail.value?.case.status ?? "");
const showCapture = computed(() =>
  ["draft", "ready", "needs_retake"].includes(status.value)
);
const showWaiting = computed(() => ["queued", "analyzing"].includes(status.value));
const showResult = computed(() =>
  ["completed", "inconclusive", "failed"].includes(status.value)
);

async function load(quiet = false): Promise<void> {
  try {
    detail.value = await api.getCase(props.id);
    loadError.value = "";
  } catch (error) {
    if (quiet) {
      // 轮询期间的单次失败：保留已加载内容，不打断等待/结果视图
      return;
    }
    loadError.value = "加载失败或无权访问";
    detail.value = null;
  } finally {
    // 首次加载（无论成败）都终止 loading，失败进入可终止的错误态
    loading.value = false;
  }
}

function ensurePolling(): void {
  if (timer) return;
  timer = setInterval(async () => {
    await load(true);
    if (!["queued", "analyzing"].includes(detail.value?.case.status ?? "")) {
      stopPolling();
    }
  }, 2500);
}

function stopPolling(): void {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

function onAnalyzing(): void {
  ensurePolling();
  void load(true);
}

onMounted(async () => {
  await load();
  if (["queued", "analyzing"].includes(detail.value?.case.status ?? "")) {
    ensurePolling();
  }
});
onUnmounted(stopPolling);
</script>

<template>
  <CaseResultView
    v-if="detail && showResult"
    :key="`result-${status}`"
    :case-id="props.id"
  />
  <section v-else-if="detail && showWaiting" class="card center">
    <p class="muted">分析排队/进行中，请稍候…（{{ status }}）</p>
    <p class="muted small">完成后自动显示结果</p>
  </section>
  <CaseCaptureView
    v-else-if="detail && showCapture"
    :key="`capture-${status}`"
    :case-id="props.id"
    @analyzing="onAnalyzing"
    @refresh="load(true)"
  />
  <div v-else-if="loading" class="muted center">加载中…</div>
  <div v-else class="card">
    <p class="error">{{ loadError || "案件不存在或无权访问。" }}</p>
    <p class="muted small">请返回案件列表重试。</p>
  </div>
</template>
