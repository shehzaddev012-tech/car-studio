#!/bin/bash
# Car Studio — one-time setup on a GCP Compute Engine VM (Ubuntu 22.04+)
# Run AFTER SSH into the VM:
#   curl -fsSL https://get.docker.com | sudo sh
#   sudo usermod -aG docker $USER && newgrp docker
#   bash gcp-vm-setup.sh

set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-wow-car-496713}"
CREDS_SRC="${GOOGLE_APPLICATION_CREDENTIALS:-./backend/wow-car-496713-8ffc25ef4315.json}"

echo "==> Installing Docker Compose plugin if missing..."
if ! docker compose version &>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y docker-compose-plugin
fi

echo "==> Writing .env..."
cat > .env <<EOF
AI_PROVIDER=compositing
GOOGLE_CLOUD_PROJECT=${PROJECT_ID}
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json
VERTEX_MIN_CONFIDENCE=0.55
LPIPS_ENABLED=true
QC_SSIM_THRESHOLD=0.85
JOB_TIMEOUT_SECONDS=600
EOF

if [[ -f "${CREDS_SRC}" ]]; then
  echo "==> GCP key found at ${CREDS_SRC}"
else
  echo "!! Upload your service account JSON to backend/ before building."
fi

echo "==> Building (first time ~10-15 min on VM, much faster than Windows)..."
docker compose build

echo "==> Starting stack..."
docker compose up -d

VM_IP=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip || echo "localhost")

echo ""
echo "Done. Open in browser:"
echo "  UI:  http://${VM_IP}:5173"
echo "  API: http://${VM_IP}:8000/health"
echo ""
echo "If port 5173 is blocked, open firewall:"
echo "  gcloud compute firewall-rules create car-studio-ui --allow tcp:5173,tcp:8000 --target-tags=car-studio"
