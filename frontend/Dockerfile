FROM node:22-alpine

WORKDIR /app

COPY frontend/package.json ./package.json
RUN npm install

COPY frontend ./

ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start"]

