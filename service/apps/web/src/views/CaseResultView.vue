<script setup lang="ts">
import { onMounted, ref, computed } from "vue";
import { useRouter } from "vue-router";
import { api, ApiError, type CaseDetail, type CaseImage, type Prediction } from "../api";
import { RESULT_DISPLAY, verdictInfo, STATUS_TEXT } from "../results";

const props = defineProps<{ caseId: string }>();
const router = useRouter();
const detail = ref<CaseDetail | null>(null);
const errorMsg = ref("");
const busy = ref("");
const reviewReason = ref("");

const outcome = computed(() =>
  verdictInfo(
    detail.value?.case.result_type ?? null,
    Boolean(detail.value?.failure),
    Boolean(detail.value?.scope_enforced),
    Boolean(detail.value?.result_verdict_valid),
    detail.value?.scope?.scope as string | null | undefined
  )
);

const wholeImages = computed(() =>
  (detail.value?.images ?? []).filter((image: CaseImage) => image.view_type === "whole")
);
const nonParticipating = computed(() =>
  (detail.value?.images ?? []).filter((image: CaseImage) => image.view_type !== "whole")
);
const activeReview = computed(() =>
  (detail.value?.review_requests ?? []).find(
    (item: Record<string, unknown>) =>
      item.status === "requested" || item.status === "in_review"
  )
);

async function load(): Promise<void> {
  try {
    detail.value = await api.getCase(props.caseId);
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "加载失败";
  }
}

async function doRequestReview(): Promise<void> {
  busy.value = "review";
  errorMsg.value = "";
  try {
    await api.requestReview(props.caseId, reviewReason.value || "请专家复核");
    await load();
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "申请失败";
  } finally {
    busy.value = "";
  }
}

async function toggleResearch(): Promise<void> {
  busy.value = "consent";
  try {
    const data = detail.value;
    if (!data) return;
    await api.updateConsent(props.caseId, !data.consent.research_training);
    await load();
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "更新失败";
  } finally {
    busy.value = "";
  }
}

async function requestDeletion(): Promise<void> {
  if (!window.confirm("确认申请删除该案件原图？")) return;
  busy.value = "delete";
  try {
    await api.requestDeletion(props.caseId);
    await load();
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "操作失败";
  } finally {
    busy.value = "";
  }
}

function newRetakeCase(): void {
  router.push({ name: "cases" });
}

function percent(value: number | undefined): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

onMounted(load);
</script>

<template>
  <section v-if="detail">
    <div class="row-between">
      <h1>鉴定结果</h1>
      <span class="chip" :class="`tone-${outcome.tone}`">{{
        STATUS_TEXT[detail.case.status] || detail.case.status
      }}</span>
    </div>

    <div v-if="detail.failure" class="card">
      <p class="error">分析失败（工程错误，无真伪结论）</p>
      <p class="muted small">错误类别：{{ detail.failure.category }}</p>
      <button class="btn" type="button" @click="newRetakeCase">新建案件复拍</button>
    </div>

    <template v-else>
      <div class="card" :class="`tone-${outcome.tone}`">
        <h2>{{ outcome.display }}</h2>
        <p class="muted small">结果类型：{{ detail.case.result_type || "—" }}</p>
        <p v-if="outcome.notice" class="notice">{{ outcome.notice }}</p>
      </div>

      <div v-if="detail.prediction && detail.result_verdict_valid" class="card">
        <h3>
          {{
            detail.result_verdict_valid
              ? "模型结论（闭集条件内）"
              : "实验预测（未通过范围门控，仅供参考，不作为真伪结论）"
          }}
        </h3>
        <p>{{ detail.prediction_message }}</p>
        <p
          v-if="detail.prediction.authenticity_label === 'counterfeit' && detail.result_verdict_valid"
          class="notice"
        >
          阶段 A 不能确认伪品的实际物种；结论为“疑似伪品”，建议人工复核。
        </p>
        <p v-else-if="!detail.result_verdict_valid" class="notice">
          范围证据缺失或图片超出范围门控，以上原始模型概率仅供参考，不作为真伪判定。
        </p>
        <div class="progress-row">
          <span>正品特征</span>
          <div class="bar">
            <span
              :style="{
                width: percent(detail.prediction.probabilities.authentic),
              }"
            ></span>
          </div>
          <span>{{ percent(detail.prediction.probabilities.authentic) }}</span>
        </div>
        <div class="progress-row">
          <span>伪品特征</span>
          <div class="bar">
            <span
              :style="{
                width: percent(detail.prediction.probabilities.counterfeit),
                background: 'var(--warning)',
              }"
            ></span>
          </div>
          <span>{{ percent(detail.prediction.probabilities.counterfeit) }}</span>
        </div>
        <p class="muted small">
          概率为闭集二分类内条件概率，不代表真实世界鉴定确定性。
        </p>
        <p class="small">
          模型版本：{{ detail.prediction.model_version }} · 策略版本：
          {{ detail.prediction.strategy_version }} · 分析时间：
          {{ new Date(detail.prediction.created_at).toLocaleString() }}
        </p>
      </div>

      <div v-if="wholeImages.length" class="card">
        <h3>参与分析的整根主图</h3>
        <div class="grid-images">
          <figure v-for="image in wholeImages" :key="image.id">
            <img :src="image.content_url" alt="whole" />
            <figcaption>{{ image.width }}×{{ image.height }}</figcaption>
          </figure>
        </div>
      </div>

      <div v-if="nonParticipating.length" class="card">
        <h3>未参与判断的视角图</h3>
        <p class="muted small">
          detail / cross_section 在阶段 A 不参与融合判断，仅用于专家复核与数据积累。
        </p>
        <div class="grid-images">
          <figure v-for="image in nonParticipating" :key="image.id">
            <img :src="image.content_url" :alt="image.view_type" />
            <figcaption>{{ image.view_type }}</figcaption>
          </figure>
        </div>
      </div>

      <div class="card disclaimer">
        {{ detail.disclaimer }} 本结果不能替代药典鉴定、显微、理化、分子检测或专业人员的结论。
      </div>

      <div class="card">
        <h3>专家复核</h3>
        <p v-if="activeReview" class="muted">
          已有复核申请：{{ (activeReview as Record<string, unknown>).status === "requested" ? "待受理" : "复核中" }}
        </p>
        <template v-else>
          <label>复核说明（可选）</label>
          <textarea v-model="reviewReason" rows="2"></textarea>
          <button class="btn" type="button" :disabled="busy === 'review'" @click="doRequestReview">
            申请专家复核
          </button>
        </template>
        <ul v-if="detail.expert_reviews.length" class="status-list">
          <li v-for="review in detail.expert_reviews" :key="String(review.id)">
            <span class="small">
              v{{ review.version }} · 真伪：{{ review.authenticity_class }} · 炮制：
              {{ review.processing_status }} · {{ review.actual_species || "物种未确认" }}
            </span>
          </li>
        </ul>
      </div>

      <div class="card">
        <label class="row-between">
          <span>研究训练授权（独立记录，可撤销）</span>
          <input type="checkbox" :checked="detail.consent.research_training" @change="toggleResearch" />
        </label>
        <button class="btn btn-secondary" type="button" @click="requestDeletion">
          申请删除原图
        </button>
        <button class="btn btn-secondary" type="button" @click="newRetakeCase">
          返回案件列表
        </button>
      </div>
    </template>

    <p v-if="errorMsg" class="error">{{ errorMsg }}</p>
  </section>
</template>
