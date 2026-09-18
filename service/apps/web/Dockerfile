# 阶段 A Web（Vue 3 + Vite + TypeScript）构建后以 nginx 静态托管
FROM node:22-alpine AS build
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm ci || npm install
COPY apps/web/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY apps/web/deploy-nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
