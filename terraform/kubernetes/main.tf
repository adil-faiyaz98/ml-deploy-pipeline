# ------------------------------------------------------------------------------
# CLOUD-AGNOSTIC KUBERNETES RESOURCES
# ------------------------------------------------------------------------------
provider "kubernetes" {
  # Configuration comes from the cloud-specific provider configuration
}

provider "helm" {
  # Configuration comes from the cloud-specific provider configuration
}

locals {
  cloud_provider = var.cloud_provider # aws, azure, or gcp
}

# ------------------------------------------------------------------------------
# NAMESPACE SETUP
# ------------------------------------------------------------------------------
resource "kubernetes_namespace" "mlops" {
  metadata {
    name = "mlops"
    
    labels = {
      environment = var.environment
    }
  }
}

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
    
    labels = {
      environment = var.environment
    }
  }
}

resource "kubernetes_namespace" "logging" {
  metadata {
    name = "logging"
    
    labels = {
      environment = var.environment
    }
  }
}

# ------------------------------------------------------------------------------
# SECRETS MANAGEMENT
# ------------------------------------------------------------------------------
resource "kubernetes_secret" "postgres_credentials" {
  metadata {
    name      = "postgres-credentials"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }
  
  data = {
    username = var.postgres_username
    password = var.postgres_password
    host     = var.postgres_host
    port     = var.postgres_port
    database = "mlmodels"
  }
}

resource "kubernetes_secret" "redis_credentials" {
  metadata {
    name      = "redis-credentials"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }
  
  data = {
    host     = var.redis_host
    password = var.redis_password
  }
}

resource "kubernetes_secret" "storage_credentials" {
  metadata {
    name      = "storage-credentials"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }
  
  data = {
    # Provider-specific storage credentials
    access_key = var.storage_access_key
    secret_key = var.storage_secret_key
    endpoint   = var.storage_endpoint
  }
}

# ------------------------------------------------------------------------------
# MLFLOW DEPLOYMENT
# ------------------------------------------------------------------------------
module "mlflow" {
  source = "../modules/helm-release"
  
  enabled      = var.deploy_mlflow
  
  release_name = "mlflow"
  chart        = "mlflow"
  repository   = "https://larribas.me/helm-charts"
  namespace    = kubernetes_namespace.mlops.metadata[0].name
  
  values = [
    file("${path.module}/values/mlflow-values.yaml"),
    file("${path.module}/values/mlflow-values-${local.cloud_provider}.yaml")
  ]
  
  set = [
    {
      name  = "backendStore.postgres.host"
      value = var.postgres_host
    },
    {
      name  = "backendStore.postgres.port"
      value = var.postgres_port
    },
    {
      name  = "backendStore.postgres.database"
      value = "mlmodels"
    },
    {
      name  = "backendStore.postgres.user"
      value = var.postgres_username
    },
    {
      name  = "env.MLFLOW_S3_ENDPOINT_URL"
      value = var.storage_endpoint
    }
  ]
  
  set_sensitive = [
    {
      name  = "backendStore.postgres.password"
      value = var.postgres_password
    }
  ]
  
  depends_on = [
    kubernetes_namespace.mlops,
    kubernetes_secret.postgres_credentials,
    kubernetes_secret.storage_credentials
  ]
}

# ------------------------------------------------------------------------------
# PROMETHEUS/GRAFANA MONITORING STACK
# ------------------------------------------------------------------------------
module "monitoring_stack" {
  source = "../modules/helm-release"
  
  enabled      = var.deploy_monitoring
  
  release_name = "kube-prometheus-stack"
  chart        = "kube-prometheus-stack"
  repository   = "https://prometheus-community.github.io/helm-charts"
  namespace    = kubernetes_namespace.monitoring.metadata[0].name
  
  values = [
    file("${path.module}/values/prometheus-values.yaml"),
    file("${path.module}/values/prometheus-values-${local.cloud_provider}.yaml")
  ]
  
  set = [
    {
      name  = "grafana.adminPassword"
      value = var.grafana_admin_password
    }
  ]
  
  depends_on = [
    kubernetes_namespace.monitoring
  ]
}

# ------------------------------------------------------------------------------
# ELASTIC STACK LOGGING
# ------------------------------------------------------------------------------
module "elastic_stack" {
  source = "../modules/helm-release"
  
  enabled      = var.deploy_elastic_stack
  
  release_name = "elastic-stack"
  chart        = "elastic-stack"
  repository   = "https://helm.elastic.co"
  namespace    = kubernetes_namespace.logging.metadata[0].name
  
  values = [
    file("${path.module}/values/elastic-stack-values.yaml"),
    file("${path.module}/values/elastic-stack-values-${local.cloud_provider}.yaml")
  ]
  
  depends_on = [
    kubernetes_namespace.logging
  ]
}