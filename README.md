# Leesin Deployment Guide

## 1. Before You Push

Check these before uploading the repo to GitHub:

- Do not upload `Lee_sin.venv/`, `.vscode/`, or other local-only folders.
- `data_cluster_store.json` is ignored by Git. It can contain saved numeric experiment-cluster vectors, so keep it out of GitHub.
- Review `goal_store.json` once. It should contain only sanitized public Goal/Axis examples.
- Remote admin writes require `ADMIN_TOKEN` on Render. Keep `ALLOW_REMOTE_ADMIN=false`.

## 2. What Is Stored

- `goal_store.json`: Experiment Goal, Axis, Domain Range, Resolution, and K_m.
- `data_cluster_store.json`: only the mapped numeric axis vector, row count, primary axis, timestamp, and goal id.
- Not stored: original uploaded files, filenames, unmapped columns, notes, operator names, emails, or other personal fields.

The stored data cluster is the target cluster's between-feature position vector. It is appended after a successful analysis and can be used as part of future Peer Groups.

## 3. Push To GitHub

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

## 4. Deploy On Render

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
- `LEESIN_STORE_DIR=/var/data`
- a persistent disk mounted at `/var/data`

When deploy finishes, Render gives you a public `onrender.com` URL that anyone can open.

During Blueprint creation, Render asks for `ADMIN_TOKEN`. Use that token in the settings dialog before saving or deleting Goals.

## 5. Important Notes

- Render provides the `PORT` environment variable automatically. This app already reads it.
- The app must bind to `0.0.0.0` on Render, and this repo is already configured for that.
- Persistent cluster/goal writes require a paid Render service with a persistent disk. Free Render web services have an ephemeral filesystem, so file changes are lost on restart/redeploy/spin-down.
- For public testing, `USE_DEMO_PEER_GROUP=true` keeps sanitized built-in Peer Groups available. Set it to `false` only after you have enough saved clusters for each Goal.
- If you want your own domain later, add it in the Render service settings after the first deploy succeeds.

## 6. Local Run

```powershell
.\run_app.ps1
```
