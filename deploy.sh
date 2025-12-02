#!/bin/bash
# FinRA - Google Cloud Run Deployment Script
# Usage: ./deploy.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 FinRA Cloud Run Deployment${NC}"
echo "=================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Install it first:${NC}"
    echo "   brew install google-cloud-sdk"
    exit 1
fi

# Check if logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Not logged in. Running: gcloud auth login${NC}"
    gcloud auth login
fi

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ No GCP project set. Run:${NC}"
    echo "   gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}📦 Project: ${PROJECT_ID}${NC}"

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY not set in environment${NC}"
    echo "   You'll need to set it in Cloud Run console or via:"
    echo "   gcloud run services update finra --set-env-vars=OPENAI_API_KEY=your-key"
fi

# Enable required APIs
echo -e "${YELLOW}📡 Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com --quiet

# Deploy using Cloud Run source deploy (builds automatically)
echo -e "${YELLOW}🔨 Building and deploying to Cloud Run...${NC}"
echo "   This may take 5-10 minutes (Playwright image is large)"

gcloud run deploy finra \
    --source . \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars="LLM_MODEL=gpt-4o-mini"

# Get the URL
SERVICE_URL=$(gcloud run services describe finra --region us-central1 --format="value(status.url)")

echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo "=================================="
echo -e "🌐 URL: ${GREEN}${SERVICE_URL}${NC}"
echo ""
echo -e "${YELLOW}⚠️  Don't forget to set your OpenAI API key:${NC}"
echo "   gcloud run services update finra --region us-central1 --set-env-vars=OPENAI_API_KEY=your-key"
echo ""
echo "📊 View logs:"
echo "   gcloud run logs read finra --region us-central1"
