#!/bin/bash

pushd ..

echo "Deleting Minikube"
minikube delete

echo "Starting Minikube"
minikube start --driver=docker

echo "Applying configurations"
kubectl create configmap mysql-core-initdb --from-file=schema.sql=database/schema/schema.sql

pushd ../notifications
kubectl create configmap mysql-notifications-initdb --from-file=schema.sql=database/schema/schema.sql

pushd ../supply
kubectl create configmap mysql-supply-initdb --from-file=schema.sql=database/schema/schema.sql

pushd ../demand
kubectl create configmap mysql-demand-initdb --from-file=schema.sql=database/schema/schema.sql

pushd ../core/kubernetes
kubectl apply -f clusterrole.yaml

echo "Applying deployments"
kubectl apply -f mysql-core.yaml
kubectl apply -f mysql-notifications.yaml
kubectl apply -f mysql-supply.yaml
kubectl apply -f mysql-demand.yaml

kubectl apply -f redis.yaml
kubectl apply -f zookeeper.yaml
kubectl apply -f kafka.yaml

kubectl apply -f gateway.yaml

kubectl apply -f chainoptim-core.yaml
kubectl apply -f chainoptim-notifications.yaml
kubectl apply -f chainoptim-supply.yaml
kubectl apply -f chainoptim-demand.yaml

kubectl create namespace monitoring

kubectl apply -f prometheus-clusterrole.yaml
kubectl apply -f prometheus-clusterrolebinding.yaml

kubectl apply -f prometheus.yaml
kubectl apply -f grafana.yaml

pushd ../../scripts
