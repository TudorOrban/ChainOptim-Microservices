FROM vault:latest

# Install jq for JSON parsing
RUN apt-get update && apt-get install -y jq

# Copy custom entrypoint script
COPY vault_entrypoint.sh /vault_entrypoint.sh
RUN chmod +x /vault_entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/vault_entrypoint.sh"]