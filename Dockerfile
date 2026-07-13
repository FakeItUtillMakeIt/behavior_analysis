FROM sevnce-registry.cn-chengdu.cr.aliyuncs.com/sevnce/anomaly:freeze1.1

WORKDIR /app
COPY . .
RUN chmod +x /app/start_service.sh

EXPOSE 10235

ENTRYPOINT ["/app/start_service.sh"]
