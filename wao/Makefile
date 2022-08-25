include .env

VERSION?=$(shell git describe --tags --match "v*")

push-image-kube-proxy: build-image-kube-proxy
	docker push $(IMAGE_REGISTRY)/wao-kube-proxy:$(VERSION)

push-image-kube-scheduler: build-image-kube-scheduler
	docker push $(IMAGE_REGISTRY)/wao-kube-scheduler:$(VERSION)

build-image-kube-proxy: kube-proxy
	docker build -f kube-proxy.dockerfile -t $(IMAGE_REGISTRY)/wao-kube-proxy:$(VERSION) .

build-image-kube-scheduler: kube-scheduler
	docker build -f kube-scheduler.dockerfile -t $(IMAGE_REGISTRY)/wao-kube-scheduler:$(VERSION) .

kube-proxy:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -mod=mod -o kube-proxy ../cmd/kube-proxy/proxy.go

kube-scheduler:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -mod=mod -o kube-scheduler ../cmd/kube-scheduler/scheduler.go

clean:
	rm -f kube-proxy
	rm -f kube-scheduler