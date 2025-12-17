variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run deployment"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "paperchat"
}

variable "openrouter_api_key" {
  description = "OpenRouter API key for LLM access"
  type        = string
  sensitive   = true
  default     = ""
}

variable "openrouter_model" {
  description = "OpenRouter model for chat"
  type        = string
  default     = "anthropic/claude-sonnet-4"
}

variable "openrouter_model_vlm" {
  description = "OpenRouter model for VLM (PDF parsing)"
  type        = string
  default     = "qwen/qwen3-vl-235b-a22b-instruct"
}

variable "docker_image_tag" {
  description = "Tag for the Docker image"
  type        = string
  default     = "latest"
}