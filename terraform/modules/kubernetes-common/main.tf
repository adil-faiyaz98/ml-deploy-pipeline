# ------------------------------------------------------------------------------
# NAMESPACE RESOURCES
# ------------------------------------------------------------------------------
resource "kubernetes_namespace" "mlops" {
  metadata {
    name = "mlops"
    
    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }
  }
}

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
    
    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }
  }
}

resource "kubernetes_namespace" "logging" {
  metadata {
    name = "logging"
    
    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }
  }
}

# ------------------------------------------------------------------------------
# NETWORK POLICIES
# ------------------------------------------------------------------------------
resource "kubernetes_network_policy" "mlops_default_deny" {
  metadata {
    name      = "default-deny"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }
  
  spec {
    pod_selector {
      match_labels = {}
    }
    
    policy_types = ["Ingress", "Egress"]
  }
}

resource "kubernetes_network_policy" "mlops_allow_api" {
  metadata {
    name      = "allow-api-traffic"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }
  
  spec {
    pod_selector {
      match_labels = {
        app = "model-api"
      }
    }
    
    ingress {
      from {
        namespace_selector {
          match_labels = {
            kubernetes.io/metadata.name = "default"
          }
        }
      }
      
      ports {
        port     = "8000"
        protocol = "TCP"
      }
      
      ports {
        port     = "8080"
        protocol = "TCP"
      }
    }
    
    egress {
      to {
        pod_selector {
          match_labels = {
            app = "mlflow"
          }
        }
      }
      
      to {
        pod_selector {
          match_labels = {
            app = "redis"
          }
        }
      }
      
      # Allow DNS resolution
      to {
        namespace_selector {}
        pod_selector {
          match_labels = {
            "k8s-app" = "kube-dns"
          }
        }
      }
      
      ports {
        port     = "53"
        protocol = "UDP"
      }
      
      ports {
        port     = "53"
        protocol = "TCP"
      }
    }
    
    policy_types = ["Ingress", "Egress"]
  }
}

# ------------------------------------------------------------------------------
# CONFIG MAPS
# ------------------------------------------------------------------------------
resource "kubernetes_config_map" "model_api_config" {
  metadata {
    name      = "model-api-config"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }
  
  data = {
    "config.json" = jsonencode({
      api = {
        host           = "0.0.0.0"
        port           = 8000
        workers        = 4
        cors_origins   = ["*"]
      }
      models = {
        dir             = "/models"
        default_version = "latest"
        auto_reload     = true
        reload_interval = 60
      }
      security = {
        enabled        = true
        api_key_header = "X-API-Key"
      }
      logging = {
        level           = "info"
        request_logging = true
      }
      performance = {
        batch_size      = 32
        cache_predictions = true
        cache_size      = 1024
      }
    })
  }
}

# ------------------------------------------------------------------------------
# SECRETS
# ------------------------------------------------------------------------------
resource "kubernetes_secret" "model_api_keys" {
  metadata {
    name      = "model-api-keys"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }
  
  data = {
    "api-keys.json" = jsonencode({
      keys = [
        var.api_key_1,
        var.api_key_2
      ]
    })
  }
  
  type = "Opaque"
}

# ------------------------------------------------------------------------------
# RESOURCE QUOTAS
# ------------------------------------------------------------------------------
resource "kubernetes_resource_quota" "mlops" {
  metadata {
    name      = "mlops-quota"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }
  
  spec {
    hard = {
      "requests.cpu"    = var.environment == "production" ? "24" : "12"
      "requests.memory" = var.environment == "production" ? "48Gi" : "24Gi"
      "limits.cpu"      = var.environment == "production" ? "48" : "24"
      "limits.memory"   = var.environment == "production" ? "96Gi" : "48Gi"
      "pods"            = "50"
    }
  }
}

# ------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------
variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
}

variable "api_key_1" {
  description = "First API key for model API"
  type        = string
  sensitive   = true
  default     = ""
}

variable "api_key_2" {
  description = "Second API key for model API"
  type        = string
  sensitive   = true
  default     = ""
}