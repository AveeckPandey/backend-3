param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [string]$Region = "asia-south1",
    [string]$ServiceName = "backend-api",
    [string]$SecretName = "gemini-api-key",
    [switch]$AllowUnauthenticated
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud CLI is not installed. Install the Google Cloud SDK first."
}

$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Setting gcloud project to $ProjectId..."
gcloud config set project $ProjectId | Out-Host

Write-Host "Enabling required services..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com | Out-Host

$secretExists = $true
gcloud secrets describe $SecretName --project $ProjectId 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    $secretExists = $false
}

if (-not $secretExists) {
    throw "Secret '$SecretName' does not exist in project '$ProjectId'. Create it first, then rerun this script."
}

$authFlag = if ($AllowUnauthenticated) { "--allow-unauthenticated" } else { "--no-allow-unauthenticated" }

Write-Host "Deploying $ServiceName to Cloud Run in $Region..."
Push-Location $repoRoot
try {
    gcloud run deploy $ServiceName `
        --source . `
        --region $Region `
        --platform managed `
        --set-secrets "GEMINI_API_KEY=$SecretName:latest" `
        --cpu 1 `
        --memory 1Gi `
        --timeout 300 `
        $authFlag | Out-Host
}
finally {
    Pop-Location
}

Write-Host "Deployment finished."
