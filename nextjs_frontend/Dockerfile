FROM public.ecr.aws/docker/library/node:20-alpine3.18

COPY . /nextjs-frontend

WORKDIR /nextjs-frontend

# アプリケーションのビルド
RUN rm -rf node_modules 
RUN rm -rf package-lock.json
RUN npm install 
RUN rm -rf .next
RUN npm run build

# アプリケーションの起動コマンド
#CMD ["npm", "start"]

#開発用のサーバー起動コマンド
CMD ["npm", "run", "dev"]