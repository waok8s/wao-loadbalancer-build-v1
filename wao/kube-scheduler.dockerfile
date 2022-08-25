FROM k8s.gcr.io/kube-scheduler:v1.19.7
COPY ./kube-scheduler /usr/local/bin/kube-scheduler
