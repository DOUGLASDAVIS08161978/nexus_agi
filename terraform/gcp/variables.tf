variable "db_password" {
  description = "PostgreSQL database password"
  type        = string
  sensitive   = true
}

variable "notification_email" {
  description = "Email for monitoring notifications"
  type        = string
  default     = "admin@example.com"
}
