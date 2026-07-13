docker run -it --name freeze_detect_v1 --restart always -v /etc/localtime:/etc/localtime:ro -p 10235:10235  sevnce-registry-vpc.cn-chengdu.cr.aliyuncs.com/sevnce/anomaly:freeze1.0  /bin/bash -c "/root/start.sh"

 docker run -it --entrypoint /bin/bash  sevnce-registry.cn-chengdu.cr.aliyuncs.com/sevnce/anomaly:freeze1.0