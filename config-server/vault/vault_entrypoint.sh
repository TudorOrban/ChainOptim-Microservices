#!/bin/bash

vault server -config=/vault/config -dev &

sleep 5

vault operator init -key-shares=1 -key-threshold=1 -format=json > /vault/init.json

UNSEAL_KEY=$(cat /vault/init.json | jq -r ".unseal_keys_b64[0]")
ROOT_TOKEN=$(cat /vault/init.json | jq -r ".root_token")

vault operator unseal $UNSEAL_KEY

kubectl create secret generic vault-secrets --from-literal=root-token=$ROOT_TOKEN --dry-run=client -o yaml | kubectl apply -f -

tail -f /dev/null