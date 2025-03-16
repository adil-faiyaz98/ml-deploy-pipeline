# ------------------------------------------------------------------------------
# KUBERNETES DEPLOYMENTS MODULE
# ------------------------------------------------------------------------------
# This module handles all Kubernetes deployments in a cloud-agnostic way

variable "kubernetes_config_path" {
  description = "Path to the kubernetes config file"
  type        = string
  default     = "~/.kube/config"
}

variable "kubernetes_config_context" {
  description = "Kubernetes config context"
  type        = string
  default     = null
}

variable "cloud_provider" {
  description = "Cloud provider (aws, azure, gcp)"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
}

variable "model_api_version" {
  description = "Version of the model API to deploy"
  type        = string
  default     = "latest"
}

variable "container_registry" {
  description = "URL of the container registry"
  type        = string
}

variable "db_host" {
  description = "Database host"
  type        = string
}

variable "db_name" {
  description = "Database name"
  type        = string
}

variable "db_user" {
  description = "Database user"
  type        = string
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "redis_host" {
  description = "Redis host"
  type        = string
}

variable "redis_password" {
  description = "Redis password"
  type        = string
  sensitive   = true
}

variable "storage_bucket" {
  description = "Storage bucket for model artifacts"
  type        = string
}

variable "domain" {
  description = "Base domain for the application"
  type        = string
  default     = "example.com"
}

variable "enable_gpu" {
  description = "Whether to enable GPU support"
  type        = bool
  default     = false
}

variable "mlflow_tracking_enabled" {
  description = "Whether to deploy MLflow tracking server"
  type        = bool
  default     = true
}

variable "monitoring_enabled" {
  description = "Whether to deploy monitoring stack"
  type        = bool
  default     = true
}

variable "logging_enabled" {
  description = "Whether to deploy logging stack"
  type        = bool
  default     = true
}

# Configure Kubernetes provider
provider "kubernetes" {
  config_path    = var.kubernetes_config_path
  config_context = var.kubernetes_config_context
}

provider "helm" {
  kubernetes {
    config_path    = var.kubernetes_config_path
    config_context = var.kubernetes_config_context
  }
}

# Get cloud-specific settings
locals {
  is_aws   = var.cloud_provider == "aws"
  is_azure = var.cloud_provider == "azure"
  is_gcp   = var.cloud_provider == "gcp"
  
  # Set cloud-specific variables
  storage_class = local.is_aws ? "gp2" : local.is_azure ? "managed-premium" : "standard"
  
  service_account_annotation_key = local.is_aws ? "eks.amazonaws.com/role-arn" : local.is_azure ? "azure.workload.identity/client-id" : "iam.gke.io/gcp-service-account"
  
  values_file_path = "${path.module}/../../kubernetes/model-deployment/values.yaml"
  
  common_labels = {
    "app.kubernetes.io/managed-by" = "terraform"
    "app.kubernetes.io/environment" = var.environment
    "cloud-provider" = var.cloud_provider
  }
}

# Create namespaces
resource "kubernetes_namespace" "ml_system" {
  metadata {
    name = "ml-system"
    
    labels = merge(local.common_labels, {
      "name" = "ml-system"
      "istio-injection" = "enabled"
    })
  }
}

resource "kubernetes_namespace" "monitoring" {
  count = var.monitoring_enabled ? 1 : 0
  
  metadata {
    name = "monitoring"
    
    labels = merge(local.common_labels, {
      "name" = "monitoring"
    })
  }
}

resource "kubernetes_namespace" "logging" {
  count = var.logging_enabled ? 1 : 0
  
  metadata {
    name = "logging"
    
    labels = merge(local.common_labels, {
      "name" = "logging"
    })
  }
}

# Create secrets
resource "kubernetes_secret" "database_credentials" {
  metadata {
    name      = "database-credentials"
    namespace = kubernetes_namespace.ml_system.metadata[0].name
  }
  
  data = {
    DB_HOST     = var.db_host
    DB_NAME     = var.db_name
    DB_USER     = var.db_user
    DB_PASSWORD = var.db_password
  }
}

resource "kubernetes_secret" "redis_credentials" {
  metadata {
    name      = "redis-credentials"
    namespace = kubernetes_namespace.ml_system.metadata[0].name
  }
  
  data = {
    REDIS_HOST     = var.redis_host
    REDIS_PASSWORD = var.redis_password
  }
}

# Create model API service account with appropriate cloud permissions
resource "kubernetes_service_account" "model_api" {
  metadata {
    name      = "model-api-sa"
    namespace = kubernetes_namespace.ml_system.metadata[0].name
    
    annotations = {
      (local.service_account_annotation_key) = var.service_account_role_arn
    }
    
    labels = local.common_labels
  }
}

# Deploy main application using Helm
resource "helm_release" "model_deployment" {
  name       = "ml-deploy"
  namespace  = kubernetes_namespace.ml_system.metadata[0].name
  chart      = "${path.module}/../../kubernetes/model-deployment"
  
  # Use values file from our Kubernetes dir
  values = [file(local.values_file_path)]
  
  # Override with our dynamic values
  set {
    name  = "global.environment"
    value = var.environment
  }
  
  set {
    name  = "global.cloudProvider"
    value = var.cloud_provider
  }
  
  set {
    name  = "global.domain"
    value = var.domain
  }
  
  set {
    name  = "global.images.registry"
    value = var.container_registry
  }
  
  set {
    name  = "modelApi.image.tag"
    value = var.model_api_version
  }
  
  set {
    name  = "modelApi.serviceAccount.annotations.${local.service_account_annotation_key}"
    value = var.service_account_role_arn
  }
  
  set {
    name  = "global.storage.bucket"
    value = var.storage_bucket
  }
  
  set {
    name  = "mlflow.enabled"
    value = var.mlflow_tracking_enabled
  }
  
  set {
    name  = "monitoring.enabled"
    value = var.monitoring_enabled
  }
  
  set {
    name  = "logging.enabled"
    value = var.logging_enabled
  }
  
  set_sensitive {
    name  = "database.password"
    value = var.db_password
  }
  
  set_sensitive {
    name  = "redis.password"
    value = var.redis_password
  }
  
  depends_on = [
    kubernetes_namespace.ml_system,
    kubernetes_secret.database_credentials,
    kubernetes_secret.redis_credentials,
    kubernetes_service_account.model_api
  ]
}

# Deploy monitoring stack if enabled
resource "helm_release" "monitoring_stack" {
  count = var.monitoring_enabled ? 1 : 0
  
  name       = "kube-prometheus-stack"
  namespace  = kubernetes_namespace.monitoring[0].metadata[0].name
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  
  values = [
    file("${path.module}/../../kubernetes/prometheus-values.yaml")
  ]
  
  depends_on = [
    kubernetes_namespace.monitoring
  ]
}

# Deploy logging stack if enabled
resource "helm_release" "logging_stack" {
  count = var.logging_enabled ? 1 : 0
  
  name       = "elastic-stack"
  namespace  = kubernetes_namespace.logging[0].metadata[0].name
  repository = "https://helm.elastic.co"
  chart      = "elastic-stack"
  
  values = [
    file("${path.module}/../../kubernetes/elastic-stack-values.yaml")
  ]
  
  depends_on = [
    kubernetes_namespace.logging
  ]
}

output "model_api_endpoint" {
  description = "Endpoint URL for the model API"
  value       = "https://api.${var.domain}"
}

output "mlflow_endpoint" {
  description = "Endpoint URL for MLflow"
  value       = var.mlflow_tracking_enabled ? "https://mlflow.${var.domain}" : null
}

output "grafana_endpoint" {
  description = "Endpoint URL for Grafana"
  value       = var.monitoring_enabled ? "https://grafana.${var.domain}" : null
}

output "kibana_endpoint" {
  description = "Endpoint URL for Kibana"
  value       = var.logging_enabled ? "https://kibana.${var.domain}" : null
}