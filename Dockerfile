FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install --production

COPY model ./model
COPY server.js ./

EXPOSE 3000

CMD ["node", "server.js"]