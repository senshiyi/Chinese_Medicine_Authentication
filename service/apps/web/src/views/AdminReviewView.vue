<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api, ApiError, type CaseImage, type Prediction } from "../api";
import { STATUS_TEXT } from "../results";

type ReviewItem = Record<string, unknown> & { id: string };

interface ExpertReviewRow {
  id: string;
  version: number;
  authenticity_class: string;
  processing_status: string;
  actual_species: string | null;
  verification_method: string;
}

interface ReviewDetailShape {
  review_request?: { id: string };
  expert_reviews?: ExpertReviewRow[];
  images?: CaseImage[];
  model_prediction?: Prediction | null;
}

const queue = ref<ReviewItem[]>([]);
const currentStatus = ref("requested");
const selected = ref<ReviewDetailShape | null>(null);
const errorMsg = ref("");
const saving = ref(false);
const keepOpen = ref(false);

const form = ref({
  actual_species: "",
  authenticity_class: "authentic",
  processing_status: "compliant",
  verification_method: "专家",
  basis: "",
});

const STATUS_OPTIONS = ["requested", "in_review", "completed"];
const AUTH_OPTIONS = ["authentic", "substitute", "adulterant", "unknown"];
const PROCESSING_OPTIONS = ["compliant", "noncompliant", "not_applicable", "unknown"];

async function loadQueue(): Promise<void> {
  errorMsg.value = "";
  try {
    queue.value = (await api.adminListReviews(currentStatus.value)) as ReviewItem[];
    selected.value = null;
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "加载队列失败";
  }
}

async function selectReview(id: string): Promise<void> {
  try {
    selected.value = (await api.adminReviewDetail(id)) as unknown as ReviewDetailShape;
    errorMsg.value = "";
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "加载详情失败";
  }
}

function selectedId(): string | null {
  return selected.value?.review_request?.id ?? null;
}

function isSelected(id: string): boolean {
  return selectedId() === id;
}

function expertReviews(): ExpertReviewRow[] {
  return selected.value?.expert_reviews ?? [];
}

function imagesOf(viewType: string): CaseImage[] {
  return (selected.value?.images ?? []).filter((image) => image.view_type === viewType);
}

function prediction(): Prediction | null {
  return selected.value?.model_prediction ?? null;
}

async function submitDecision(): Promise<void> {
  saving.value = true;
  errorMsg.value = "";
  try {
    const reviewId = selectedId();
    if (!reviewId) {
      throw new ApiError(400, "no_review", "未选择复核申请");
    }
    await api.adminDecision(reviewId, {
      actual_species: form.value.actual_species || null,
      authenticity_class: form.value.authenticity_class,
      processing_status: form.value.processing_status,
      verification_method: form.value.verification_method,
      basis: form.value.basis || null,
      keep_open: keepOpen.value,
    });
    await selectReview(reviewId);
    await loadQueue();
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "提交失败";
  } finally {
    saving.value = false;
  }
}

async function switchStatus(value: string): Promise<void> {
  currentStatus.value = value;
  await loadQueue();
}

onMounted(loadQueue);
</script>

<template>
  <section>
    <h1>专家复核</h1>
    <div class="row-between" style="margin: 10px 0">
      <button
        v-for="option in STATUS_OPTIONS"
        :key="option"
        class="btn btn-secondary"
        style="width: auto"
        type="button"
        :class="{ active: currentStatus === option }"
        @click="switchStatus(option)"
      >
        {{ option }}
      </button>
    </div>
    <p v-if="errorMsg" class="error">{{ errorMsg }}</p>

    <div class="row-between" style="align-items: flex-start">
      <div style="flex: 1 1 220px">
        <div
          v-for="item in queue"
          :key="item.id"
          class="card review-item"
          :class="{ selected: isSelected(String(item.id)) }"
          @click="selectReview(String(item.id))"
        >
          <div class="row-between">
            <strong>#{{ String(item.id).slice(0, 8) }}</strong>
            <span class="chip">{{ STATUS_TEXT[String(item.case_status)] || item.case_status }}</span>
          </div>
          <div class="muted small">请求于 {{ new Date(String(item.requested_at)).toLocaleString() }}</div>
          <div v-if="item.reason" class="small">原因：{{ item.reason }}</div>
        </div>
        <div v-if="!queue.length" class="card muted">该状态暂无复核申请。</div>
      </div>

      <div v-if="selected" style="flex: 2 1 460px; min-width: 0">
        <div class="card">
          <h3>多视角图片</h3>
          <h4 class="small">整根（参与分析）</h4>
          <div class="grid-images">
            <figure v-for="image in imagesOf('whole')" :key="image.id">
              <img :src="image.content_url" alt="whole" />
              <figcaption>{{ image.width }}×{{ image.height }}</figcaption>
            </figure>
          </div>
          <h4 class="small">局部 / 断面（未参与分析）</h4>
          <div class="grid-images">
            <figure
              v-for="image in [...imagesOf('detail'), ...imagesOf('cross_section')]"
              :key="image.id"
            >
              <img :src="image.content_url" :alt="image.view_type" />
              <figcaption>{{ image.view_type }}</figcaption>
            </figure>
          </div>

          <div v-if="prediction()" class="card small">
            模型：{{ prediction()?.authenticity_label }}（
            {{ ((prediction()?.confidence ?? 0) * 100).toFixed(1) }}%）· 版本
            {{ prediction()?.model_version }} · 实验闭集：是
          </div>

          <h4>历史专家版本</h4>
          <ul class="status-list">
            <li v-for="review in expertReviews()" :key="review.id">
              v{{ review.version }} · {{ review.authenticity_class }} ·
              {{ review.processing_status }} · {{ review.actual_species || "物种未确认" }} ·
              {{ review.verification_method }}
            </li>
            <li v-if="!expertReviews().length" class="muted">暂无专家结论</li>
          </ul>
        </div>

        <div class="card">
          <h3>录入专家结论（追加版本）</h3>
          <label>实际物种（actual_species，可空）</label>
          <input v-model="form.actual_species" type="text" maxlength="160" placeholder="如：槲蕨" />
          <label>真伪类别</label>
          <select v-model="form.authenticity_class">
            <option v-for="item in AUTH_OPTIONS" :key="item" :value="item">{{ item }}</option>
          </select>
          <label>炮制状态</label>
          <select v-model="form.processing_status">
            <option v-for="item in PROCESSING_OPTIONS" :key="item" :value="item">{{ item }}</option>
          </select>
          <label>鉴定依据方法（verification_method）*</label>
          <input v-model="form.verification_method" type="text" maxlength="160" required />
          <label>依据说明（可选）</label>
          <textarea v-model="form.basis" rows="3"></textarea>
          <label class="row-between">
            <span>保持未结案以便继续追加</span>
            <input v-model="keepOpen" type="checkbox" />
          </label>
          <p v-if="errorMsg" class="error">{{ errorMsg }}</p>
          <button class="btn" type="button" :disabled="saving" @click="submitDecision">
            {{ saving ? "提交中…" : "提交专家结论" }}
          </button>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.review-item {
  cursor: pointer;
}
.review-item.selected {
  border-color: var(--primary);
}
.active {
  border-color: var(--primary);
  background: #e7f2ec;
  color: var(--primary-dark);
}
</style>
