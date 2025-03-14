variable "enabled" {
  description = "Whether to create the Helm release"
  type        = bool
  default     = true
}

variable "release_name" {
  description = "Name of the Helm release"
  type        = string
}

variable "chart" {
  description = "Chart name to be installed"
  type        = string
}

variable "repository" {
  description = "Repository URL where to locate the chart"
  type        = string
  default     = null
}

variable "namespace" {
  description = "The namespace to install the release into"
  type        = string
  default     = "default"
}

variable "create_namespace" {
  description = "Create the namespace if it does not yet exist"
  type        = bool
  default     = false
}

variable "chart_version" {
  description = "Specify the exact chart version to install"
  type        = string
  default     = null
}

variable "values" {
  description = "List of values in raw yaml to pass to helm"
  type        = list(string)
  default     = []
}

variable "set" {
  description = "Value block with custom values to be merged with the values yaml"
  type        = list(object({
    name  = string
    value = string
  }))
  default     = []
}

variable "set_sensitive" {
  description = "Value block with custom sensitive values to be merged with the values yaml"
  type        = list(object({
    name  = string
    value = string
  }))
  default     = []
}

variable "timeout" {
  description = "Time in seconds to wait for any individual kubernetes operation"
  type        = number
  default     = 300
}

variable "atomic" {
  description = "If set, installation process purges chart on fail"
  type        = bool
  default     = true
}

variable "cleanup_on_fail" {
  description = "Allow deletion of new resources created in this upgrade when upgrade fails"
  type        = bool
  default     = true
}

variable "wait" {
  description = "Will wait until all resources are in a ready state before marking the release as successful"
  type        = bool
  default     = true
}

variable "recreate_pods" {
  description = "Perform pods restart during upgrade/rollback"
  type        = bool
  default     = false
}

variable "max_history" {
  description = "Maximum number of release versions stored per release"
  type        = number
  default     = 10
}

resource "helm_release" "this" {
  count = var.enabled ? 1 : 0

  name       = var.release_name
  chart      = var.chart
  repository = var.repository
  version    = var.chart_version
  namespace  = var.namespace
  
  create_namespace = var.create_namespace
  
  values = var.values
  
  dynamic "set" {
    for_each = var.set
    content {
      name  = set.value.name
      value = set.value.value
    }
  }
  
  dynamic "set_sensitive" {
    for_each = var.set_sensitive
    content {
      name  = set_sensitive.value.name
      value = set_sensitive.value.value
    }
  }
  
  wait             = var.wait
  atomic           = var.atomic
  cleanup_on_fail  = var.cleanup_on_fail
  timeout          = var.timeout
  recreate_pods    = var.recreate_pods
  max_history      = var.max_history
}

output "name" {
  description = "Name of the release"
  value       = var.enabled ? helm_release.this[0].name : null
}

output "version" {
  description = "Version of the release"
  value       = var.enabled ? helm_release.this[0].version : null
}

output "namespace" {
  description = "Namespace of the release"
  value       = var.enabled ? helm_release.this[0].namespace : null
}

output "status" {
  description = "Status of the release"
  value       = var.enabled ? helm_release.this[0].status : null
}