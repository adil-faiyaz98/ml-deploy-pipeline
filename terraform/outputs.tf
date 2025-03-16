# ------------------------------------------------------------------------------
# MULTI-CLOUD OUTPUTS
# ------------------------------------------------------------------------------

output "kubernetes_cluster_name" {
  description = "Name of the Kubernetes cluster"
  value = try(
    module.aws[0].eks_cluster_id,
    module.azure[0].aks_cluster_name,
    module.gcp[0].gke_cluster_name,
    null
  )
}

output "kubernetes_endpoint" {
  description = "Endpoint for the Kubernetes API"
  value = try(
    module.aws[0].eks_cluster_endpoint,
    module.azure[0].aks_cluster_endpoint,
    module.gcp[0].gke_cluster_endpoint,
    null
  )
  sensitive = true
}

output "database_endpoint" {
  description = "Endpoint for the database"
  value = try(
    module.aws[0].db_instance_endpoint,
    module.azure[0].db_server_fqdn,
    module.gcp[0].db_instance_ip,
    null
  )
  sensitive = true
}

output "model_storage_name" {
  description = "Storage name for model artifacts"
  value = try(
    module.aws[0].model_artifacts_bucket,
    module.azure[0].model_storage_account_name,
    module.gcp[0].model_bucket_name,
    null
  )
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value = try(
    module.aws[0].redis_endpoint,
    module.azure[0].redis_host,
    module.gcp[0].redis_endpoint,
    null
  )
  sensitive = true
}

output "cloud_provider" {
  description = "Active cloud provider"
  value       = var.cloud_provider
}

output "deployed_environment" {
  description = "Deployed environment"
  value       = var.environment
}