FROM sevnce-registry.cn-chengdu.cr.aliyuncs.com/sevnce/anomaly:freeze1.1

WORKDIR /app

# 安装依赖（使用清华镜像源加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制代码
COPY . .
RUN chmod +x /app/start_service.sh

EXPOSE 10235

ENTRYPOINT ["/app/start_service.sh"]
