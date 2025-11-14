# Terraform Configuration for Nexus AGI on Google Cloud Platform

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "nexus-agi-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
  ])
  
  service = each.key
  disable_on_destroy = false
}

# VPC Network
resource "google_compute_network" "nexus_vpc" {
  name                    = "nexus-agi-vpc"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.required_apis]
}

resource "google_compute_subnetwork" "nexus_subnet" {
  name          = "nexus-agi-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.nexus_vpc.id
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# GKE Cluster
resource "google_container_cluster" "nexus_cluster" {
  name     = "nexus-agi-cluster"
  location = var.zone
  
  network    = google_compute_network.nexus_vpc.name
  subnetwork = google_compute_subnetwork.nexus_subnet.name
  
  remove_default_node_pool = true
  initial_node_count       = 1
  
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
  }
  
  depends_on = [google_compute_subnetwork.nexus_subnet]
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "primary-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.nexus_cluster.name
  node_count = 3
  
  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }
  
  node_config {
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
    
    labels = {
      environment = var.environment
      app         = "nexus-agi"
    }
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# Cloud SQL PostgreSQL
resource "google_sql_database_instance" "nexus_postgres" {
  name             = "nexus-postgres"
  database_version = "POSTGRES_14"
  region           = var.region
  
  settings {
    tier = "db-f1-micro"
    
    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }
    
    ip_configuration {
      ipv4_enabled    = true
      private_network = google_compute_network.nexus_vpc.id
      
      authorized_networks {
        name  = "all"
        value = "0.0.0.0/0"
      }
    }
    
    database_flags {
      name  = "max_connections"
      value = "100"
    }
  }
  
  deletion_protection = false
  depends_on          = [google_project_service.required_apis]
}

resource "google_sql_database" "nexus_db" {
  name     = "nexus"
  instance = google_sql_database_instance.nexus_postgres.name
}

resource "google_sql_user" "nexus_user" {
  name     = "nexus"
  instance = google_sql_database_instance.nexus_postgres.name
  password = var.db_password
}

# Cloud Storage Buckets
resource "google_storage_bucket" "nexus_storage" {
  name          = "${var.project_id}-nexus-storage"
  location      = var.region
  force_destroy = false
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
}

resource "google_storage_bucket" "nexus_backups" {
  name          = "${var.project_id}-nexus-backups"
  location      = var.region
  force_destroy = false
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
}

# Service Accounts
resource "google_service_account" "nexus_sa" {
  account_id   = "nexus-agi-sa"
  display_name = "Nexus AGI Service Account"
}

resource "google_service_account" "aria_sa" {
  account_id   = "aria-sa"
  display_name = "ARIA Service Account"
}

# IAM Bindings
resource "google_project_iam_member" "nexus_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.nexus_sa.email}"
}

resource "google_project_iam_member" "nexus_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.nexus_sa.email}"
}

resource "google_project_iam_member" "nexus_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.nexus_sa.email}"
}

# Secret Manager
resource "google_secret_manager_secret" "github_token" {
  secret_id = "github-token"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret" "huggingface_token" {
  secret_id = "huggingface-token"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret" "openai_api_key" {
  secret_id = "openai-api-key"
  
  replication {
    automatic = true
  }
}

# Cloud Monitoring
resource "google_monitoring_notification_channel" "email" {
  display_name = "Nexus AGI Email Notifications"
  type         = "email"
  
  labels = {
    email_address = var.notification_email
  }
}

resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "High CPU Usage - Nexus AGI"
  combiner     = "OR"
  
  conditions {
    display_name = "CPU usage above 80%"
    
    condition_threshold {
      filter          = "resource.type = \"k8s_container\" AND resource.labels.namespace_name = \"nexus-agi\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.id]
}

# Cloud Build Trigger
resource "google_cloudbuild_trigger" "nexus_build" {
  name        = "nexus-agi-build"
  description = "Build and deploy Nexus AGI on push to main"
  
  github {
    owner = "DOUGLASDAVIS08161978"
    name  = "nexus_agi"
    
    push {
      branch = "^main$"
    }
  }
  
  filename = "cloudbuild.yaml"
}

# Outputs
output "gke_cluster_name" {
  value       = google_container_cluster.nexus_cluster.name
  description = "GKE cluster name"
}

output "gke_cluster_endpoint" {
  value       = google_container_cluster.nexus_cluster.endpoint
  description = "GKE cluster endpoint"
  sensitive   = true
}

output "postgres_connection_name" {
  value       = google_sql_database_instance.nexus_postgres.connection_name
  description = "Cloud SQL connection name"
}

output "storage_bucket" {
  value       = google_storage_bucket.nexus_storage.name
  description = "Main storage bucket name"
}

output "nexus_service_account" {
  value       = google_service_account.nexus_sa.email
  description = "Nexus service account email"
}

output "aria_service_account" {
  value       = google_service_account.aria_sa.email
  description = "ARIA service account email"
}
