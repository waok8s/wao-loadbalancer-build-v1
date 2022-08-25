# Build WAO container images

## 1. Compile binaries

```sh
make kube-proxy
make kube-scheduler
```

## 2. (Optional) Set your container registory

Edit `.env`.

```sh
IMAGE_REGISTRY=hoge.example.com
```

## 3. Build container images

```sh
make build-image-kube-proxy
make build-image-kube-scheduler
```

## 4. (Optional) Push images to your container registory

```sh
make push-image-kube-proxy
make push-image-kube-scheduler
```
