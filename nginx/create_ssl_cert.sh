#!/bin/bash

# Create ssl directory if it doesn't exist
mkdir -p ssl

# Generate a self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/server.key -out ssl/server.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
  -addext "subjectAltName = DNS:localhost,IP:127.0.0.1"

# Set proper permissions
chmod 600 ssl/server.key
chmod 644 ssl/server.crt

echo "Self-signed SSL certificate generated successfully."
echo "Key: ssl/server.key"
echo "Certificate: ssl/server.crt" 