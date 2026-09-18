<script setup lang="ts">
import { defineProps } from "vue";
import type { CaseDetail } from "../api";
import { STATUS_TEXT } from "../results";

const props = defineProps<{ detail: CaseDetail | null; loading: boolean }>();
defineEmits<{ (e: "go-result"): void }>();
</script>

<template>
  <section>
    <div v-if="loading" class="muted center">加载中…</div>
    <div v-else-if="detail" class="center">
      <p class="muted">
        正在等待分析结果（案件 {{ detail.case.id.slice(0, 8) }}）
      </p>
      <p>
        <span class="chip">{{
          STATUS_TEXT[detail.case.status] || detail.case.status
        }}</span>
      </p>
      <p class="muted small">完成后将自动跳转到结果页…</p>
    </div>
  </section>
</template>
