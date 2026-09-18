import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: "*.spec.ts",
  timeout: 60_000,
  retries: 0,
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://localhost:8080",
  },
});
