# Leesin Deployment Guide

## 1. Before You Push

Check these before uploading the repo to GitHub:

- Do not upload `Lee_sin.venv/`, `.vscode/`, or other local-only folders.
- Review `goal_store.json` once. If it contains real internal experiment names or sensitive setup values, replace them with sanitized examples before pushing.
- Remote admin access is disabled by default. Keep `ALLOW_REMOTE_ADMIN=false` unless you add authentication first.

## 2. Push To GitHub

If you have not created a GitHub repo yet:

1. Create a new empty repository on GitHub.
2. In this project folder, run:

```powershell
git init
git add .
git commit -m "Initial Leesin deployment"
git branch -M main
git remote add origin https://github.com/<your-account>/<your-repo>.git
git push -u origin main
```

If this project is already a Git repo, just run:

```powershell
git add .
git commit -m "Prepare Leesin for Render"
git push
```

## 3. Deploy On Render

This repo already includes a root-level `render.yaml`, so the easiest path is Blueprint deploy.

1. Log in to Render.
2. Connect your GitHub account.
3. In the Render dashboard, click `New +` -> `Blueprint`.
4. Select this repository.
5. Confirm the linked branch, usually `main`.
6. Deploy the Blueprint.

Render will read:

- `runtime: python`
- `buildCommand: pip install -r requirements.txt`
- `startCommand: python Leesin.py --host 0.0.0.0`
- `healthCheckPath: /health`

When deploy finishes, Render gives you a public `onrender.com` URL that anyone can open.

## 4. Important Notes

- Render provides the `PORT` environment variable automatically. This app already reads it.
- The app must bind to `0.0.0.0` on Render, and this repo is already configured for that.
- `goal_store.json` is file-based. Changes made from a running Render instance are not durable across restarts/redeploys unless you move that data to a database or persistent storage.
- If you want your own domain later, add it in the Render service settings after the first deploy succeeds.

## 5. Local Run

```powershell
.\run_app.ps1
```
