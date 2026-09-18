<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api, ApiError, type CaseDetail, type CaseImage } from "../api";
import { STATUS_TEXT } from "../results";

const props = defineProps<{ caseId: string }>();
const emit = defineEmits<{ (e: "analyzing"): void; (e: "refresh"): void }>();

const detail = ref<CaseDetail | null>(null);
const errorMsg = ref("");
const uploading = ref<Record<string, boolean>>({});
const submitting = ref(false);

const VIEW_LABEL: Record<string, { title: string; hint: string }> = {
  whole: { title: "整根主图", hint: "参与本次实验性分析（必传一张质量合格）" },
  detail: { title: "局部纹理图", hint: "仅用于专家复核与数据积累，不参与分析" },
  cross_section: { title: "断面图", hint: "仅用于专家复核与数据积累，不参与分析" },
};

async function load(): Promise<void> {
  try {
    detail.value = await api.getCase(props.caseId);
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "加载失败";
  }
}

async function handleFile(view: string, event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files && input.files[0];
  if (!file) return;
  errorMsg.value = "";
  uploading.value[view] = true;
  try {
    await api.uploadImage(props.caseId, view, file);
    await load();
    emit("refresh");
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "上传失败";
  } finally {
    uploading.value[view] = false;
    input.value = "";
  }
}

async function removeImage(image: CaseImage): Promise<void> {
  errorMsg.value = "";
  try {
    await api.deleteImage(props.caseId, image.id);
    await load();
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "删除失败";
  }
}

async function startAnalyze(): Promise<void> {
  errorMsg.value = "";
  submitting.value = true;
  try {
    await api.analyze(props.caseId);
    emit("analyzing");
  } catch (error) {
    errorMsg.value = error instanceof ApiError ? error.message : "提交分析失败";
  } finally {
    submitting.value = false;
  }
}

onMounted(load);
</script>

<template>
  <section v-if="detail">
    <div class="row-between">
      <h1>拍摄与上传</h1>
      <span class="chip">{{ STATUS_TEXT[detail.case.status] || detail.case.status }}</span>
    </div>

    <div v-if="detail.case.status === 'needs_retake'" class="notice">
      主图质量不足（模糊/曝光问题），请重新拍摄并上传合格的整根主图后再分析。
    </div>
    <div v-else class="notice">{{ detail.experimental_notice }}</div>

    <div v-for="(meta, view) in VIEW_LABEL" :key="view" class="card" :data-testid="`upload-${view}`">
      <h3>{{ meta.title }}</h3>
      <p class="muted small">{{ meta.hint }}</p>
      <div class="grid-images">
        <figure
          v-for="image in detail.images.filter((item) => item.view_type === view)"
          :key="image.id"
        >
          <img :src="image.content_url" :alt="view" />
          <figcaption>
            {{ image.width }}×{{ image.height }}
            <span v-if="image.quality_flags.length" class="chip tone-warning">
              {{ image.quality_flags.join("、") }}
            </span>
          </figcaption>
        </figure>
      </div>
      <div class="file-row">
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          :disabled="uploading[view]"
          @change="handleFile(view, $event)"
        />
        <button class="btn btn-secondary" style="width: auto" type="button" :disabled="uploading[view]">
          {{ uploading[view] ? "上传中…" : "上传" }}
        </button>
      </div>
      <button
        v-for="image in detail.images.filter((item) => item.view_type === view)"
        :key="`del-${image.id}`"
        class="btn btn-secondary"
        style="width: auto"
        type="button"
        @click="removeImage(image)"
      >
        删除该图
      </button>
      <p class="muted small">支持 JPEG/PNG/WebP；≤20MB；≤4000万像素。</p>
    </div>

    <p v-if="errorMsg" class="error">{{ errorMsg }}</p>

    <div class="card">
      <button
        class="btn"
        type="button"
        :disabled="detail.case.status !== 'ready' || submitting"
        :data-testid="`analyze-btn-${detail.case.status}`"
        @click="startAnalyze"
      >
        {{ submitting ? "提交中…" : "开始实验性分析" }}
      </button>
      <p class="muted small center">
        仅整根主图参与分析；detail 与 cross_section 只用于复核与数据积累。
      </p>
    </div>
  </section>
</template>
