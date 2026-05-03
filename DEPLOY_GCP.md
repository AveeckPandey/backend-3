# Deploy FastAPI Backend to GCP

This backend is set up to deploy to **Google Cloud Run**.

## 1. Install and authenticate the Google Cloud CLI

On Windows, install the Google Cloud SDK first. One option is:

```powershell
winget install -e --id Google.CloudSDK --accept-source-agreements --accept-package-agreements
```

If the installer prompts for administrator approval, accept it and let the installation finish.

Then open a fresh terminal and run:

```powershell
gcloud auth login
gcloud auth application-default login
```

## 2. Pick your project and region

Example values:

- Project ID: `your-gcp-project-id`
- Region: `asia-south1`
- Service name: `backend-api`

## 3. Create the Gemini API secret in Secret Manager

Do this once per project:

```powershell
gcloud config set project your-gcp-project-id
gcloud services enable secretmanager.googleapis.com
'YOUR_GEMINI_API_KEY' | gcloud secrets create gemini-api-key --data-file=-
```

If the secret already exists and you want to rotate the value:

```powershell
'YOUR_GEMINI_API_KEY' | gcloud secrets versions add gemini-api-key --data-file=-
```

## 4. Deploy the service

From this repo:

```powershell
.\scripts\deploy-gcp.ps1 -ProjectId your-gcp-project-id -Region asia-south1 -ServiceName backend-api -AllowUnauthenticated
```

That command:

- enables the required GCP APIs
- deploys from source to Cloud Run
- injects `GEMINI_API_KEY` from Secret Manager
- sets the container runtime for FastAPI automatically via the included `Dockerfile`
- uploads only the intended deployment files because `.gcloudignore` excludes local secrets, reports, notebooks, and virtualenv files

## 5. Test the deployment

Cloud Run will print a service URL. Check health:

```powershell
curl https://YOUR_CLOUD_RUN_URL/health
```

Expected response:

```json
{"status":"ok"}
```

## Notes

- The container listens on `PORT`, which Cloud Run provides automatically.
- `reports/` is ephemeral on Cloud Run. Generated PDFs will work for the current instance lifetime, but they are not durable storage. If you want persistent report downloads, move generated reports to Google Cloud Storage.
- Do not store API keys in `.env.local` for production deployments. Use Secret Manager instead.
