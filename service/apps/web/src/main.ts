import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";
import "./styles.css";

createApp(App).use(router).mount("#app");

if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // PWA 离线能力是增强项，注册失败不影响使用
    });
  });
}
