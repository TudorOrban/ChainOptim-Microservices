FROM vault:latest

# Install jq for JSON parsing
RUN apt-get update && apt-get install -y jq

COPY vault_entrypoint.sh /vault_entrypoint.sh
RUN chmod +x /vault_entrypoint.sh

ENTRYPOINT ["/vault_entrypoint.sh"]