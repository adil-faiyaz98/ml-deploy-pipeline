name: Multi-Cloud ML Pipeline CI/CD---------------------------------------------
# REQUIRED PARAMETERS
on:-----------------------------------------------------------------------------
  push:
    branches: [ main ]
    paths:ion = "The project ID to deploy to"
      - 'src/**'string
      - 'terraform/**'
      - 'deployment/**'
      - 'kubernetes/**'
      - '.github/workflows/**' "GCP region"
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:--------------------------------------------------------------
      environment:
        description: 'Environment to deploy to'----------------------------------------------------------
        required: true
        default: 'dev'me" {
        type: choice"Name of the project"
        options: string
          - dev
          - staging
          - production" {
      cloud_provider:tion)"
        description: 'Cloud provider to deploy to'
        required: true
        default: 'aws'
        type: choice {
        options: "Owner of the resources"
          - awstring
          - azure
          - gcp
able "cost_center" {
env: center for billing"
  TF_VERSION: '1.5.7'
  PYTHON_VERSION: '3.10'
  TF_IN_AUTOMATION: 'true'
  ARM_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}---------------------------------------------------------------------------
  ARM_CLIENT_SECRET: ${{ secrets.AZURE_CLIENT_SECRET }}S PARAMETERS
  ARM_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}--------------------------------------------------------------
  ARM_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
kubernetes_version" {
jobs:etes version"
  validate:
    name: Validate  default     = "1.28"
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code_nodes" {
        uses: actions/checkout@v3e pool"
  type        = number
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'ription = "Maximum number of nodes in GKE node pool"

      - name: Install dependencies= 10
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txtvariable "desired_nodes" {
ber of nodes in GKE node pool"
      - name: Lint Python code= number
        run: |
          flake8 src/}
          black --check src/

      - name: Run tests = "GKE node machine type"
        run: |
          pytest src/tests/ --cov=src    = "e2-standard-4"

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with: nodes"
          terraform_version: ${{ env.TF_VERSION }}
          se
      - name: Validate Terraform
        run: |
          cd terraformhine_type" {
          for dir in aws azure gcp; do= "GPU node machine type"
            echo "Validating $dir..."  type        = string
            cd $dirt     = "n1-standard-8"
            terraform init -backend=false
            terraform validate
            cd ..
          done
    = string
  build:la-t4"
    name: Build and Push Container
    needs: validate
    if: github.event_name == 'push' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:  type        = number
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Docker Buildx# ------------------------------------------------------------------------------
        uses: docker/setup-buildx-action@v2
----------------------------------------------------------------
      - name: Get short SHA
        id: sha
        run: echo "short=$(git rev-parse --short HEAD)" >> $GITHUB_OUTPUT

      - name: Set environment variables= "db-custom-4-16384"
        id: vars
        run: |
          if [ "${{ github.event_name }}" == "workflow_dispatch" ]; then_version" {
            echo "environment=${{ github.event.inputs.environment }}" >> $GITHUB_OUTPUT  description = "PostgreSQL version"
            echo "cloud_provider=${{ github.event.inputs.cloud_provider }}" >> $GITHUB_OUTPUT
          else
            echo "environment=dev" >> $GITHUB_OUTPUT
            echo "cloud_provider=aws" >> $GITHUB_OUTPUT
backup_enabled" {
      # AWS Container Registry Login
      - name: Configure AWS credentials (AWS)
        if: ${{ steps.vars.outputs.cloud_provider == 'aws' }}
        uses: aws-actions/configure-aws-credentials@v1}
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}me for database backups in UTC"
          aws-region: ${{ secrets.AWS_REGION }}
  default     = "02:00"
      - name: Login to Amazon ECR (AWS)
        if: ${{ steps.vars.outputs.cloud_provider == 'aws' }}
        id: login-ecr-aws-----------------
        uses: aws-actions/amazon-ecr-login@v1
-------------------------------------------------------------------
      # Azure Container Registry Login
      - name: Login to Azure (Azure)variable "redis_node_type" {
        if: ${{ steps.vars.outputs.cloud_provider == 'azure' }}
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      - name: Login to ACR (Azure)---------------------------------
        if: ${{ steps.vars.outputs.cloud_provider == 'azure' }}
        id: login-ecr-azure# ------------------------------------------------------------------------------
        run: |
          az acr login --name ${{ secrets.AZURE_REGISTRY_NAME }}
          echo "registry=${{ secrets.AZURE_REGISTRY_NAME }}.azurecr.io" >> $GITHUB_OUTPUT

      # GCP Container Registry Login = "ml-deploy-network"
      - name: Setup gcloud CLI (GCP)
        if: ${{ steps.vars.outputs.cloud_provider == 'gcp' }}
        uses: google-github-actions/setup-gcloud@v0.6.0
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: ${{ secrets.GCP_PROJECT_ID }}  default     = "ml-deploy-subnet"
          export_default_credentials: true

      - name: Configure Docker for GCR (GCP)
        if: ${{ steps.vars.outputs.cloud_provider == 'gcp' }}  description = "CIDR range for the subnet"
        run: |
          gcloud auth configure-docker gcr.io
          echo "registry=gcr.io/${{ secrets.GCP_PROJECT_ID }}" >> $GITHUB_ENV

      # Set registry URL based on cloud provider
      - name: Set registry URL
        id: registry
        run: |
          if [ "${{ steps.vars.outputs.cloud_provider }}" == "aws" ]; then
            echo "url=${{ steps.login-ecr-aws.outputs.registry }}" >> $GITHUB_OUTPUT
          elif [ "${{ steps.vars.outputs.cloud_provider }}" == "azure" ]; thend_range_cidr" {
            echo "url=${{ steps.login-ecr-azure.outputs.registry }}" >> $GITHUB_OUTPUT  description = "CIDR range for pods"
          elif [ "${{ steps.vars.outputs.cloud_provider }}" == "gcp" ]; then
            echo "url=gcr.io/${{ secrets.GCP_PROJECT_ID }}" >> $GITHUB_OUTPUT
          fi

      # Build and push imagesname" {
      - name: Build and push model-apiange for services"
        uses: docker/build-push-action@v4ng
        with:svc-range"
          context: .
          file: ./deployment/api/Dockerfile
          push: truer" {
          tags: |
            ${{ steps.registry.outputs.url }}/model-api:latest
            ${{ steps.registry.outputs.url }}/model-api:${{ steps.sha.outputs.short }}
            ${{ steps.registry.outputs.url }}/model-api:${{ steps.vars.outputs.environment }}}
          build-args: |
            BUILD_ENV=${{ steps.vars.outputs.environment }}--------------------------------
          cache-from: type=registry,ref=${{ steps.registry.outputs.url }}/model-api:buildcache
          cache-to: type=registry,ref=${{ steps.registry.outputs.url }}/model-api:buildcache,mode=max-------------------------------------------------------------------

      - name: Build and push model-trainer
        uses: docker/build-push-action@v4t of CIDR blocks that can access the API server"
        with:ist(object({
          context: .
          file: ./deployment/trainer/Dockerfile
          push: true
          tags: |
            ${{ steps.registry.outputs.url }}/model-trainer:latest
            ${{ steps.registry.outputs.url }}/model-trainer:${{ steps.sha.outputs.short }}
            ${{ steps.registry.outputs.url }}/model-trainer:${{ steps.vars.outputs.environment }}ivate_cluster" {
          build-args: |tion = "Whether to create a private cluster"
            BUILD_ENV=${{ steps.vars.outputs.environment }}
          cache-from: type=registry,ref=${{ steps.registry.outputs.url }}/model-trainer:buildcachetrue
          cache-to: type=registry,ref=${{ steps.registry.outputs.url }}/model-trainer:buildcache,mode=max

  deploy-infrastructure:enable_workload_identity" {
    name: Deploy Infrastructure enable workload identity"
    needs: build
    if: github.event_name == 'push' || github.event_name == 'workflow_dispatch'  default     = true
    runs-on: ubuntu-latest
    steps:
      - name: Checkout codeyption_key_ring" {
        uses: actions/checkout@v3

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2        with:          terraform_version: ${{ env.TF_VERSION }}      - name: Set environment variables        id: vars        run: |          if [ "${{ github.event_name }}" == "workflow_dispatch" ]; then            echo "environment=${{ github.event.inputs.environment }}" >> $GITHUB_OUTPUT            echo "cloud_provider=${{ github.event.inputs.cloud_provider }}" >> $GITHUB_OUTPUT          else            echo "environment=dev" >> $GITHUB_OUTPUT            echo "cloud_provider=aws" >> $GITHUB_OUTPUT      # Cloud-specific credential setup      - name: Configure AWS credentials (AWS)        if: ${{ steps.vars.outputs.cloud_provider == 'aws' }}        uses: aws-actions/configure-aws-credentials@v1        with:          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}          aws-region: ${{ secrets.AWS_REGION }}      - name: Login to Azure (Azure)        if: ${{ steps.vars.outputs.cloud_provider == 'azure' }}        uses: azure/login@v1        with:          creds: ${{ secrets.AZURE_CREDENTIALS }}      - name: Setup gcloud CLI (GCP)        if: ${{ steps.vars.outputs.cloud_provider == 'gcp' }}        uses: google-github-actions/setup-gcloud@v0.6.0        with:          service_account_key: ${{ secrets.GCP_SA_KEY }}          project_id: ${{ secrets.GCP_PROJECT_ID }}          export_default_credentials: true      # Deploy infrastructure      - name: Terraform Init        working-directory: terraform        run: |          terraform init      - name: Terraform Plan        working-directory: terraform        run: |          terraform plan \            -var="cloud_provider=${{ steps.vars.outputs.cloud_provider }}" \            -var="environment=${{ steps.vars.outputs.environment }}" \            -out=tfplan      - name: Terraform Apply        working-directory: terraform        run: |          terraform apply -auto-approve tfplan  deploy-application:    name: Deploy Application    needs: deploy-infrastructure    runs-on: ubuntu-latest    steps:      - name: Checkout code        uses: actions/checkout@v3      - name: Set environment variables        id: vars        run: |          if [ "${{ github.event_name }}" == "workflow_dispatch" ]; then            echo "environment=${{ github.event.inputs.environment }}" >> $GITHUB_OUTPUT            echo "cloud_provider=${{ github.event.inputs.cloud_provider }}" >> $GITHUB_OUTPUT          else            echo "environment=dev" >> $GITHUB_OUTPUT            echo "cloud_provider=aws" >> $GITHUB_OUTPUT      # Configure cloud provider CLI for Kubernetes access      - name: Configure AWS credentials and get kubeconfig (AWS)        if: ${{ steps.vars.outputs.cloud_provider == 'aws' }}        uses: aws-actions/configure-aws-credentials@v1        with:          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}          aws-region: ${{ secrets.AWS_REGION }}      - name: Get EKS kubeconfig (AWS)        if: ${{ steps.vars.outputs.cloud_provider == 'aws' }}        run: |          aws eks update-kubeconfig --name ml-deploy-${{ steps.vars.outputs.environment }}-cluster --region ${{ secrets.AWS_REGION }}      - name: Login to Azure and get kubeconfig (Azure)        if: ${{ steps.vars.outputs.cloud_provider == 'azure' }}        uses: azure/login@v1        with:          creds: ${{ secrets.AZURE_CREDENTIALS }}      - name: Get AKS kubeconfig (Azure)        if: ${{ steps.vars.outputs.cloud_provider == 'azure' }}        run: |          az aks get-credentials --resource-group ml-deploy-${{ steps.vars.outputs.environment }}-rg --name ml-deploy-${{ steps.vars.outputs.environment }}-aks      - name: Setup gcloud CLI and get kubeconfig (GCP)        if: ${{ steps.vars.outputs.cloud_provider == 'gcp' }}        uses: google-github-actions/setup-gcloud@v0.6.0        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: ${{ secrets.GCP_PROJECT_ID }}
          export_default_credentials: true

      - name: Get GKE kubeconfig (GCP)
        if: ${{ steps.vars.outputs.cloud_provider == 'gcp' }}
        run: |
          gcloud container clusters get-credentials ml-deploy-${{ steps.vars.outputs.environment }}-cluster --region ${{ secrets.GCP_REGION }}

      # Deploy Kubernetes resources
      - name: Setup Helm
        uses: azure/setup-helm@v3
        with:
          version: 'v3.11.2'

      # Use image SHA for deterministic deployments
      - name: Get short SHA
        id: sha
        run: echo "short=$(git rev-parse --short HEAD)" >> $GITHUB_OUTPUT

      # Set registry URL based on cloud provider
      - name: Set registry URL
        id: registry
        run: |
          if [ "${{ steps.vars.outputs.cloud_provider }}" == "aws" ]; then
            echo "url=${{ secrets.AWS_ECR_REGISTRY }}" >> $GITHUB_OUTPUT
          elif [ "${{ steps.vars.outputs.cloud_provider }}" == "azure" ]; then
            echo "url=${{ secrets.AZURE_REGISTRY_NAME }}.azurecr.io" >> $GITHUB_OUTPUT
          elif [ "${{ steps.vars.outputs.cloud_provider }}" == "gcp" ]; then
            echo "url=gcr.io/${{ secrets.GCP_PROJECT_ID }}" >> $GITHUB_OUTPUT
          fi

      - name: Deploy with Helm
        run: |
          cd kubernetes/model-deployment
          
          # Update values.yaml with environment-specific settings
          sed -i "s|\${CONTAINER_REGISTRY}|${{ steps.registry.outputs.url }}|g" values.yaml
          sed -i "s|environment: dev|environment: ${{ steps.vars.outputs.environment }}|g" values.yaml
          sed -i "s|cloudProvider: aws|cloudProvider: ${{ steps.vars.outputs.cloud_provider }}|g" values.yaml
          sed -i "s|tag: latest|tag: ${{ steps.sha.outputs.short }}|g" values.yaml
          
          # Validate manifest
          helm template . --debug
          
          # Install/upgrade release
          helm upgrade --install ml-deploy . \
            --namespace ml-deploy \
            --create-namespace \
            --values values.yaml \
            --set global.environment=${{ steps.vars.outputs.environment }} \
            --set global.cloudProvider=${{ steps.vars.outputs.cloud_provider }} \
            --set modelApi.image.tag=${{ steps.sha.outputs.short }}

      - name: Verify deployment
        run: |
          kubectl rollout status deployment/model-api -n ml-deploy --timeout=5m
          kubectl get pods,services,ingress -n ml-deploy