# ------------------------------------------------------------------------------
# AZURE CONFIGURATION
# ------------------------------------------------------------------------------
provider "azurerm" {
  features {}
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  location    = var.azure_region

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Terraform   = "true"
    Owner       = var.owner
    CostCenter  = var.cost_center
  }

  aks_node_pools = {
    ml_compute = {
      name                  = "mlcompute"
      vm_size               = "Standard_D8s_v3"
      node_count            = var.desired_nodes
      min_count             = var.min_nodes
      max_count             = var.max_nodes
      enable_auto_scaling   = true
      max_pods              = 110
      orchestrator_version  = var.kubernetes_version
      os_disk_size_gb       = 100
      os_disk_type          = "Managed"
      priority              = "Regular" # Use "Spot" for dev
      labels = {
        "workload-type" = "ml-training"
      }
    },
    api_serving = {
      name                  = "apiserving"
      vm_size               = "Standard_D4s_v3"
      node_count            = 2
      min_count             = 2
      max_count             = 10
      enable_auto_scaling   = true
      max_pods              = 50
      orchestrator_version  = var.kubernetes_version
      os_disk_size_gb       = 50
      os_disk_type          = "Managed"
      priority              = "Regular"
      labels = {
        "workload-type" = "model-serving"
      }
    },
    monitoring = {
      name                  = "monitoring"
      vm_size               = "Standard_D2s_v3"
      node_count            = 1
      min_count             = 1
      max_count             = 3
      enable_auto_scaling   = true
      max_pods              = 30
      orchestrator_version  = var.kubernetes_version
      os_disk_size_gb       = 50
      os_disk_type          = "Managed"
      priority              = "Regular"
      labels = {
        "workload-type" = "monitoring"
      }
    }
  }
}

# ------------------------------------------------------------------------------
# RESOURCE GROUP
# ------------------------------------------------------------------------------
resource "azurerm_resource_group" "main" {
  name     = "${local.name_prefix}-rg"
  location = local.location
  tags     = local.common_tags
}

# ------------------------------------------------------------------------------
# VIRTUAL NETWORK
# ------------------------------------------------------------------------------
module "network" {
  source              = "../modules/azure/network"
  name                = "${local.name_prefix}-vnet"
  resource_group_name = azurerm_resource_group.main.name
  location            = local.location
  address_space       = ["10.0.0.0/16"]
  
  subnets = {
    private = {
      name           = "private"
      address_prefix = "10.0.1.0/24"
    }
    public = {
      name           = "public"
      address_prefix = "10.0.2.0/24"
    }
    db = {
      name           = "database"
      address_prefix = "10.0.3.0/24"
      service_endpoints = ["Microsoft.Sql"]
    }
  }
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# AZURE KUBERNETES SERVICE
# ------------------------------------------------------------------------------
module "aks" {
  source                 = "../modules/azure/aks"
  name                   = "${local.name_prefix}-aks"
  resource_group_name    = azurerm_resource_group.main.name
  location               = local.location
  kubernetes_version     = var.kubernetes_version
  dns_prefix             = "${local.name_prefix}-aks"
  
  # Network configuration
  vnet_subnet_id         = module.network.subnet_ids["private"]
  service_cidr           = "10.0.4.0/24"
  dns_service_ip         = "10.0.4.10"
  docker_bridge_cidr     = "172.17.0.1/16"
  
  # System node pool configuration
  default_node_pool = {
    name                = "system"
    vm_size             = "Standard_D2s_v3"
    availability_zones  = ["1", "2", "3"]
    node_count          = 1
    min_count           = 1
    max_count           = 3
    enable_auto_scaling = true
    max_pods            = 30
    os_disk_size_gb     = 50
    os_disk_type        = "Managed"
    node_labels         = { "role" = "system" }
    vnet_subnet_id      = module.network.subnet_ids["private"]
  }
  
  # Additional node pools
  node_pools = local.aks_node_pools
  
  # Authentication
  role_based_access_control_enabled = true
  admin_group_object_ids            = var.admin_group_object_ids
  
  # Add-ons
  network_policy                    = "calico"
  azure_policy_enabled              = true
  oms_agent_enabled                 = true
  key_vault_secrets_provider_enabled = true
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# AZURE DATABASE FOR POSTGRESQL
# ------------------------------------------------------------------------------
module "postgres" {
  source                = "../modules/azure/postgres"
  name                  = "${local.name_prefix}-postgres"
  resource_group_name   = azurerm_resource_group.main.name
  location              = local.location
  
  sku_name              = var.db_instance_type
  storage_mb            = 102400  # 100GB
  backup_retention_days = var.environment == "production" ? 35 : 7
  geo_redundant_backup  = var.environment == "production" ? "Enabled" : "Disabled"
  
  administrator_login    = "postgres"
  administrator_password = random_password.postgres.result
  
  version                = "11"
  ssl_enforcement_enabled = true
  
  # Network
  subnet_id              = module.network.subnet_ids["db"]
  private_endpoint_enabled = true
  
  # Database
  database_name          = "mlmodels"
  charset                = "UTF8"
  collation              = "en_US.UTF8"
  
  postgresql_configurations = {
    "max_connections" = "500"
    "shared_buffers"  = "4096MB"
    "work_mem"        = "64MB"
  }
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# AZURE REDIS CACHE
# ------------------------------------------------------------------------------
module "redis" {
  source                = "../modules/azure/redis"
  name                  = "${local.name_prefix}-redis"
  resource_group_name   = azurerm_resource_group.main.name
  location              = local.location
  
  sku_name              = var.redis_node_type
  family                = var.environment == "production" ? "P" : "C"
  capacity              = var.environment == "production" ? 2 : 1
  
  enable_non_ssl_port   = false
  minimum_tls_version   = "1.2"
  
  redis_configuration = {
    maxmemory_policy = "allkeys-lru"
    maxfragmentationmemory_reserved = "50"
    maxmemory_reserved = "50"
  }
  
  private_endpoint_enabled = true
  subnet_id = module.network.subnet_ids["private"]
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# AZURE STORAGE ACCOUNT FOR MODEL ARTIFACTS
# ------------------------------------------------------------------------------
module "storage" {
  source                = "../modules/azure/storage"
  name                  = "${local.name_prefix}storage"  # Note: name must be globally unique and lowercase
  resource_group_name   = azurerm_resource_group.main.name
  location              = local.location
  account_tier          = "Standard"
  account_replication_type = "ZRS"  # Zone-redundant storage
  
  network_rules = {
    default_action = "Deny"
    ip_rules       = var.management_ips
    subnet_ids     = [module.network.subnet_ids["private"]]
  }
  
  containers = {
    "model-artifacts" = {
      access_type = "private"
    },
    "logs" = {
      access_type = "private"
    }
  }
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------
# RANDOM PASSWORDS
# ------------------------------------------------------------------------------
resource "random_password" "postgres" {
  length  = 16
  special = false
}

resource "random_password" "redis" {
  length  = 16
  special = false
}

# ------------------------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------------------------
output "azure_resource_group" {
  description = "Azure Resource Group"
  value       = azurerm_resource_group.main.name
}

output "aks_cluster_name" {
  description = "AKS cluster name"
  value       = module.aks.cluster_name
}

output "aks_host" {
  description = "AKS cluster API server hostname"
  value       = module.aks.host
  sensitive   = true
}

output "postgres_fqdn" {
  description = "PostgreSQL FQDN"
  value       = module.postgres.fqdn
  sensitive   = true
}

output "storage_account_name" {
  description = "Storage Account Name"
  value       = module.storage.name
}

output "redis_host" {
  description = "Redis hostname"
  value       = module.redis.hostname
  sensitive   = true
}

# ------------------------------------------------------------------------------
# AZURE IMPLEMENTATION
# ------------------------------------------------------------------------------

provider "azurerm" {
  features {}
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  location    = var.location
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Terraform   = "true"
    Owner       = var.owner
    CostCenter  = var.cost_center
  }
  
  # Machine type mappings
  vm_sizes = {
    "small"  = "Standard_D2s_v3"
    "medium" = "Standard_D4s_v3"
    "large"  = "Standard_D8s_v3"
    "xlarge" = "Standard_D16s_v3"
    "gpu-small" = "Standard_NC4as_T4_v3"
    "gpu-medium" = "Standard_NC8as_T4_v3"
    "gpu-large" = "Standard_NC16as_T4_v3"
  }
  
  # Database SKU mappings
  db_skus = {
    "small"  = "GP_Gen5_2"
    "medium" = "GP_Gen5_4"
    "large"  = "GP_Gen5_8"
    "xlarge" = "GP_Gen5_16"
  }
  
  # Redis SKU mappings
  redis_skus = {
    "small"  = "Basic"
    "medium" = "Standard"
    "large"  = "Premium"
  }
  
  # Redis family mappings
  redis_families = {
    "small"  = "C"
    "medium" = "C"
    "large"  = "P"
  }
  
  # Redis capacity mappings
  redis_capacities = {
    "small"  = 1
    "medium" = 2
    "large"  = 3
  }
}

# ------------------------------------------------------------------------------
# RESOURCE GROUP
# ------------------------------------------------------------------------------
resource "azurerm_resource_group" "main" {
  name     = "${local.name_prefix}-rg"
  location = local.location
  tags     = local.common_tags
}

# ------------------------------------------------------------------------------
# NETWORKING
# ------------------------------------------------------------------------------
resource "azurerm_virtual_network" "main" {
  name                = "${local.name_prefix}-vnet"
  address_space       = [var.vnet_cidr]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

resource "azurerm_subnet" "aks" {
  name                 = "aks-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.aks_subnet_cidr]
  
  # Required for network policies
  enforce_private_link_endpoint_network_policies = false
  enforce_private_link_service_network_policies  = false
}

resource "azurerm_subnet" "db" {
  name                 = "db-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.db_subnet_cidr]
  
  # For private endpoints  
  enforce_private_link_endpoint_network_policies = true
  service_endpoints    = ["Microsoft.Sql"]
}

resource "azurerm_subnet" "redis" {
  name                 = "redis-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.redis_subnet_cidr]
  
  enforce_private_link_endpoint_network_policies = true
  service_endpoints    = ["Microsoft.Cache"]
}

# ------------------------------------------------------------------------------
# KUBERNETES CLUSTER (AKS)
# ------------------------------------------------------------------------------
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${local.name_prefix}-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${local.name_prefix}-k8s"
  kubernetes_version  = var.kubernetes_version
  
  # Use system-assigned managed identity
  identity {
    type = "SystemAssigned"
  }
  
  # Enable Azure Policy for Kubernetes
  azure_policy_enabled = true
  
  # Default node pool (for system services)
  default_node_pool {
    name                = "system"
    vm_size             = local.vm_sizes["medium"]
    enable_auto_scaling = true
    min_count           = 1
    max_count           = 3
    node_count          = 1
    vnet_subnet_id      = azurerm_subnet.aks.id
    os_disk_size_gb     = 50
    
    tags = local.common_tags
  }
  
  # Azure AD integration
  azure_active_directory_role_based_access_control {
    managed                = true
    admin_group_object_ids = var.admin_group_ids
    azure_rbac_enabled     = true
  }
  
  # Network configuration
  network_profile {
    network_plugin    = "azure"
    network_policy    = "calico"
    load_balancer_sku = "standard"
  }
  
  # Auto-upgrade settings
  automatic_channel_upgrade = var.environment == "production" ? "stable" : "patch"
  
  # Maintenance window (for production)
  dynamic "maintenance_window" {
    for_each = var.environment == "production" ? [1] : []
    content {
      allowed {
        day   = "Saturday"
        hours = [21, 22, 23, 0, 1, 2, 3, 4]
      }
      allowed {
        day   = "Sunday"
        hours = [21, 22, 23, 0, 1, 2, 3, 4]
      }
    }
  }
  
  tags = local.common_tags
}

# ML training node pool
resource "azurerm_kubernetes_cluster_node_pool" "ml" {
  name                  = "mlcompute"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.use_gpu ? local.vm_sizes["gpu-medium"] : local.vm_sizes["large"]
  
  enable_auto_scaling = true
  min_count           = var.min_nodes
  max_count           = var.max_nodes
  node_count          = var.desired_nodes
  
  os_disk_size_gb     = 100
  os_type             = "Linux"
  vnet_subnet_id      = azurerm_subnet.aks.id
  
  # Use spot instances for cost savings (except in production)
  priority        = var.environment == "production" ? "Regular" : "Spot"
  eviction_policy = var.environment == "production" ? null : "Delete"
  spot_max_price  = var.environment == "production" ? null : -1
  
  node_labels = {
    "workload-type" = "ml-training"
    "environment"   = var.environment
  }
  
  node_taints = var.use_gpu ? ["nvidia.com/gpu=present:NoSchedule"] : []
  
  tags = local.common_tags
}

# API serving node pool
resource "azurerm_kubernetes_cluster_node_pool" "api" {
  name                  = "apiserving"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = local.vm_sizes["medium"]
  
  enable_auto_scaling = true
  min_count           = 2
  max_count           = 10
  node_count          = 2
  
  os_disk_size_gb = 50
  os_type         = "Linux"
  vnet_subnet_id  = azurerm_subnet.aks.id
  
  node_labels = {
    "workload-type" = "model-serving"
    "environment"   = var.environment
  }
  
  tags = local.common_tags
}

# Monitoring node pool
resource "azurerm_kubernetes_cluster_node_pool" "monitoring" {
  name                  = "monitoring"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = local.vm_sizes["small"]
  
  enable_auto_scaling = true
  min_count           = 1
  max_count           = 3
  node_count          = 1
  
  os_disk_size_gb = 50
  os_type         = "Linux"
  vnet_subnet_id  = azurerm_subnet.aks.id
  
  node_labels = {
    "workload-type" = "monitoring"
    "environment"   = var.environment
  }
  
  tags = local.common_tags
}

# ------------------------------------------------------------------------------