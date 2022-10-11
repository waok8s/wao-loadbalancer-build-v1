include .env

VERSION?=$(shell git describe --tags --match "v*")

push-image-kube-proxy: build-image-kube-proxy
	docker push $(IMAGE_REGISTRY)/wao-kube-proxy:$(VERSION)

build-image-kube-proxy: kube-proxy
	docker build -f kube-proxy.dockerfile -t $(IMAGE_REGISTRY)/wao-kube-proxy:$(VERSION) .

kube-proxy:
	go mod edit -require=tensorflow@v0.0.0 -replace=tensorflow=./pkg/tensorflow
	go mod edit -require=tensorflow_serving@v0.0.0 -replace=tensorflow_serving=./pkg/tensorflow_serving
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -mod=mod -o kube-proxy ../cmd/kube-proxy/proxy.go

clean:
	rm -f kube-proxy