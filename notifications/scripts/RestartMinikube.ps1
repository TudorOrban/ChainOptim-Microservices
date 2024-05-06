Push-Location -Path ".."

Write-Host "Deleting Minikube"
minikube delete

Write-Host "Starting Minikube"
minikube start --driver=docker

Write-Host "Applying deployments"
kubectl apply -f clusterrole.yaml
kubectl apply -f mysql-core-config-map.yaml
kubectl apply -f mysql-notifications-config-map.yaml

kubectl apply -f mysql-core.yaml
kubectl apply -f mysql-notifications.yaml

kubectl apply -f redis.yaml
kubectl apply -f zookeeper.yaml
kubectl apply -f kafka.yaml

kubectl apply -f api-gateway.yaml

kubectl apply -f chainoptim-core.yaml
kubectl apply -f chainoptim-notifications.yaml
