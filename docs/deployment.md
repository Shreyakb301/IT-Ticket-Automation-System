# Single-Service Deployment Guide

This project is configured for one Docker deployment. The same service serves:

- React dashboard at `/`
- FastAPI docs at `/docs`
- API endpoints at `/predict`, `/analytics`, and `/health`

Recommended host: Render Docker Web Service.

## 1. Create Model Artifact Zip

The trained model folders are ignored by git because they are large. Create one zip containing the runtime artifacts:

```bash
zip -r model_artifacts.zip transformer_models models \
  -x "models/final_artifacts/*" "models/.DS_Store" "transformer_models/**/.DS_Store"
```

Upload `model_artifacts.zip` somewhere the deployed app can download it, such as a GitHub Release asset or cloud storage file URL.

The zip should extract into the project root with one or both folders:

```text
transformer_models/
models/
```

The backend uses `transformer_models/` first. If those artifacts are missing, it falls back to `models/`.

## 2. Deploy On Render

Create a new Render service:

```text
New + -> Web Service
Runtime: Docker
Repository: this GitHub repo
Branch: main
```

Render will use the root `Dockerfile`.

The Dockerfile:

1. Builds the React frontend.
2. Installs Python dependencies.
3. Copies the built frontend into `frontend/dist`.
4. Starts FastAPI with Uvicorn.

## 3. Environment Variables

Set this in Render:

```text
MODEL_ARTIFACT_URL=https://your-download-url/model_artifacts.zip
```

Optional:

```text
CORS_ORIGINS=*
```

For one-service deployment, the frontend and backend use the same domain, so CORS is not important.

Do not set `VITE_API_URL` for this deployment. In production, the frontend calls the same origin automatically.

## 4. Verify Deployment

After Render deploys, test:

```text
https://your-render-app.onrender.com
https://your-render-app.onrender.com/health
https://your-render-app.onrender.com/docs
```

Expected `/health` response:

```json
{"status":"ok"}
```

## Important Notes

- Free Render instances may be slow to start.
- Transformer artifacts are large. If startup or disk limits are a problem, upload only the smaller `models/` folder and let the app use the XGBoost fallback.
- Do not commit `models/` or `transformer_models/` directly to git unless you intentionally use Git LFS.
