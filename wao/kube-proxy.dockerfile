FROM registry.k8s.io/kube-proxy:v1.25.2
COPY ./kube-proxy /usr/local/bin/kube-proxy
