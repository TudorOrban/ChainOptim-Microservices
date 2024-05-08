Push-Location -Path ".."

Write-Host "Deleting Minikube"
minikube delete

Write-Host "Starting Minikube"
minikube start --driver=docker

Write-Host "Applying configurations"
kubectl apply -f clusterrole.yaml

kubectl create configmap mysql-notifications-initdb --from-file=schema.sql=database/schema/schema.sql
Push-Location -Path "../core"
kubectl create configmap mysql-core-initdb --from-file=schema.sql=database/schema/schema.sql
Push-Location -Path "../supply"
kubectl create configmap mysql-supply-initdb --from-file=schema.sql=database/schema/schema.sql

Write-Host "Applying deployments"
kubectl apply -f mysql-core.yaml
kubectl apply -f mysql-notifications.yaml

kubectl apply -f redis.yaml
kubectl apply -f zookeeper.yaml
kubectl apply -f kafka.yaml

kubectl apply -f api-gateway.yaml

kubectl apply -f chainoptim-core.yaml
kubectl apply -f chainoptim-notifications.yaml

kubectl apply -f prometheus-clusterrole.yaml
kubectl apply -f prometheus-clusterrolebinding.yaml

kubectl apply -f prometheus.yaml
kubectl apply -f grafana.yaml

Push-Location -Path "../notifications/scripts"