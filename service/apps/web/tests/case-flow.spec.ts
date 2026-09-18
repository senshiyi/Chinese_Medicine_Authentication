/**
 * 端到端冒烟（可选）：需要先 `docker compose -f infra/docker-compose.yml up -d --build`，
 * 前端 dev/build 与 API 经 nginx 提供后运行：
 *   cd apps/web && npm i -D @playwright/test && npx playwright install chromium
 *   npm run test:e2e
 * 未安装 @playwright/test 或未启动栈时本文件不会自动运行。
 */
import { expect, test } from "@playwright/test";

const BASE = process.env.E2E_BASE_URL || "http://localhost:8080";
const UNIQUE = `e2e-${Date.now()}@example.com`;

test("登录 → 新建案件 → 上传 whole → 分析 → 查看实验范围文案", async ({ page }) => {
  await page.goto(BASE + "/login");

  await page.getByText("注册", { exact: true }).click();
  await page.locator('input[type="email"]').fill(UNIQUE);
  await page.locator('input[type="password"]').fill("password123");
  await page.getByRole("button", { name: "注册并登录" }).click();

  await expect(page).toHaveURL(/\/cases$/);
  await page.getByRole("button", { name: "新建案件" }).click();

  await expect(page).toHaveURL(/\/cases\/[0-9a-f-]+/);
  await expect(page.getByText("整根主图")).toBeVisible();

  // 上传本地 fixture 图片（无真实 fixture 时跳过该步）
  const fixture = process.env.E2E_WHOLE_FIXTURE;
  if (fixture) {
    await page.locator('[data-testid="upload-whole"] input[type="file"]').setInputFiles(fixture);
    await expect(page.getByText("开始实验性分析")).toBeVisible({ timeout: 10000 });
    await page.getByRole("button", { name: "开始实验性分析" }).click();
    await expect(page.getByText("专家复核")).toBeVisible({ timeout: 30000 });
    await expect(page.getByText("阶段A实验")).toBeVisible();
  } else {
    test.skip(true, "缺少 E2E_WHOLE_FIXTURE，跳过分析链路（无需真实模型权重）");
  }
});
