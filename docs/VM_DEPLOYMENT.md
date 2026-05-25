# Kellner backend — VM deployment & ops pipeline

Production runs on an **Azure Linux VM** with **nginx** (public HTTP) and **systemd** (persistent FastAPI/uvicorn). Secrets live in **`.env` on the VM only** — never committed to git.

| Item | Value |
|------|--------|
| **Repo** | [github.com/yasharthpaliwal-99/kellner-be](https://github.com/yasharthpaliwal-99/kellner-be) |
| **Deploy branch** | `yash-be-push` (or `main` if that is your release branch) |
| **App path on VM** | `/opt/kellner` |
| **Linux user** | `kellner-admin` |
| **Public host** | `74.249.2.119` (replace if VM IP changes) |
| **Backend (internal)** | `127.0.0.1:8000` |
| **Public API base** | `http://74.249.2.119` |

---

## Architecture

```text
Browser / FE  →  http://74.249.2.119:80
                      ↓
                 nginx (systemd: nginx.service)
                      ↓ proxy
                 uvicorn @ 127.0.0.1:8000  (systemd: kellner.service)
                      ↓
              FastAPI app (/opt/kellner)
                      ↓
         Azure Postgres, MongoDB, Azure OpenAI, Speech, Blob, etc.
```

- **nginx** survives SSH disconnect and VM reboot (if enabled).
- **Backend must run under systemd** — not as a foreground `uvicorn` in an SSH session (that dies on disconnect or Ctrl+C).

---

## One-time VM setup

### 1. SSH access (from Mac)

```bash
chmod 400 ~/Downloads/kellner.pem
ssh -i ~/Downloads/kellner.pem kellner-admin@74.249.2.119
```

Use your real key path and VM public IP.

### 2. OS packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git nginx
```

Prefer **Python 3.11 or 3.12** for the app venv.

### 3. Clone app

```bash
sudo mkdir -p /opt/kellner
sudo chown kellner-admin:kellner-admin /opt/kellner
cd /opt/kellner
git clone https://github.com/yasharthpaliwal-99/kellner-be.git .
git checkout yash-be-push
```

If the repo already exists, skip clone and use `git pull` (see [Deploy pipeline](#deploy-pipeline-code-update) below).

### 4. Python virtualenv + dependencies

```bash
cd /opt/kellner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt gunicorn
```

Use **`python -m pip`** and **`python -m uvicorn`** so install and run share the same interpreter.

### 5. Environment file (from laptop, not git)

On your **Mac** (not inside SSH):

```bash
scp -i ~/Downloads/kellner.pem \
  /Users/yasharthpaliwal/zeel/kellner/.env \
  kellner-admin@74.249.2.119:/opt/kellner/.env
```

Copy `.env.example` locally first and fill Azure/Postgres/Mongo keys. Re-run `scp` whenever secrets or endpoints change.

Required groups of variables (see `.env.example`):

- Azure Speech, Azure OpenAI (chat + TTS), Embeddings  
- `PGSQL_*` (Azure PostgreSQL)  
- `MONGODB_URI`, `MONGODB_DB_NAME`  
- Blob storage (menu images)  
- `DEFAULT_HOTEL_ID`, optional CORS (`CORS_EXTRA_ORIGINS=http://74.249.2.119`)

**Postgres note:** default database on Azure is usually `postgres`, not the server name. Set `PGSQL_DB_NAME=postgres` unless you created another DB.

**Azure firewall:** allow the VM’s outbound IP (and developers’ IPs for local `setup_db.py`).

### 6. Database bootstrap (once per environment)

On VM (or locally with same `.env` pointing at Azure):

```bash
cd /opt/kellner
source .venv/bin/activate
python scripts/setup_db.py
python scripts/embed_menu.py
```

### 7. nginx reverse proxy

Site config (paths may vary; common on Ubuntu):

```bash
sudo nano /etc/nginx/sites-available/kellner
```

Example:

```nginx
server {
    listen 80;
    server_name 74.249.2.119;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

Enable and reload:

```bash
sudo ln -sf /etc/nginx/sites-available/kellner /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl reload nginx
```

If the SPA is served separately, you may have additional `root` / static rules; keep **`/api` and WebSocket** proxied to port 8000.

### 8. systemd service (persistent backend)

Create `/etc/systemd/system/kellner.service`:

```bash
sudo tee /etc/systemd/system/kellner.service >/dev/null <<'EOF'
[Unit]
Description=Kellner FastAPI Backend
After=network.target

[Service]
User=kellner-admin
Group=kellner-admin
WorkingDirectory=/opt/kellner
Environment="PATH=/opt/kellner/.venv/bin"
ExecStart=/opt/kellner/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
```

- **`RestartSec=3`** = wait 3 seconds **after a crash**, then restart — **not** a restart every 3 seconds while dining.
- Do **not** run a second `uvicorn` in SSH while this service is active (port 8000 conflict).

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable kellner
sudo systemctl start kellner
systemctl status kellner --no-pager
```

### 9. Smoke test

On VM:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1/health
```

From laptop (after `exit` SSH — proves prod survives disconnect):

```bash
curl -s http://74.249.2.119/health
```

Expect: `{"status":"ok"}`.

Open API docs: `http://74.249.2.119/docs`.

---

## Deploy pipeline (code update)

### A. Developer machine — push to GitHub

```bash
cd /path/to/kellner   # local clone; remote often named `backend`
git checkout yash-be-push
git add …
git commit -m "…"
git push backend yash-be-push
```

Do not commit `.env`.

### B. VM — pull and restart service

SSH in, then:

```bash
cd /opt/kellner
git fetch origin
git checkout yash-be-push
git pull origin yash-be-push
source .venv/bin/activate
python -m pip install -r requirements.txt
sudo systemctl restart kellner
```

Verify:

```bash
git rev-parse --short HEAD
systemctl status kellner --no-pager
curl -s http://127.0.0.1:8000/health
```

If `.env` changed locally, re-run **`scp`** from Mac before restart.

### C. What **not** to do on deploy

| Avoid | Why |
|--------|-----|
| Only `sudo systemctl reload nginx` | Reloads nginx only; **does not** restart Python or load new code |
| `uvicorn` in SSH foreground | Dies on SSH close or Ctrl+C |
| `git pull` without `systemctl restart kellner` | Old process keeps running old code |

---

## Production API endpoints (frontend)

| Use | URL |
|-----|-----|
| **Base** | `http://74.249.2.119` |
| **Health** | `GET /health` |
| **Device login** | `POST /api/device/login` |
| **Device session** | `GET /api/device/session` (header `x-device-session`) |
| **Kitchen board** | `GET /api/kitchen?hotel_id=<id>&date=YYYY-MM-DD` (date optional) |
| **Table order ops** | `POST /api/kitchen/order-ops` (draft order: requests + spice) |
| **Kitchen line status** | `POST /api/kitchen/line-status` (device session header) |
| **Menu** | `POST /api/fetch_menu`, `POST /api/save_menu` (see menu routes) |
| **Voice WebSocket** | `ws://74.249.2.119/api/ws/conversation?session_id=<uuid>` |

After TLS: use `https://` / `wss://` and set `CORS_EXTRA_ORIGINS` to the public origin.

Structured voice payloads: see [FRONTEND_STRUCTURED_REPLIES.md](./FRONTEND_STRUCTURED_REPLIES.md).

---

## Day-2 operations

### Restart backend

```bash
sudo systemctl restart kellner
```

### Logs

```bash
journalctl -u kellner -f --no-pager
sudo tail -f /var/log/nginx/error.log
```

### Stop / start

```bash
sudo systemctl stop kellner
sudo systemctl start kellner
```

### Check what listens on 8000

```bash
ss -lntp | grep 8000
```

### Port 8000 already in use

```bash
sudo fuser -k 8000/tcp
sudo systemctl start kellner
```

---

## Troubleshooting

### `502 Bad Gateway` on public URL

- nginx is up; **backend is down**.
- On VM: `curl http://127.0.0.1:8000/health` — if fail, `sudo systemctl start kellner` or fix crash via `journalctl -u kellner`.

### `Unit kellner.service not found`

- Service file never created — run [§8 systemd](#8-systemd-service-persistent-backend) once.

### Prod worked until SSH closed / Ctrl+C

- App was running in SSH, not systemd — set up `kellner.service` and use `systemctl restart` for deploys.

### `POST /api/kitchen/order-ops` → 404

- **Route missing:** VM on old git revision — `git pull` + `systemctl restart kellner`; confirm route in `/docs`.
- **Handler 404:** no draft order for `hotel_id` + `table_number` in Mongo (JSON body includes `detail`).

### `git pull` says up to date but code seems old

- Wrong branch: `git branch` → should be `yash-be-push` (or your deploy branch).
- Compare: `git rev-parse HEAD` vs latest on GitHub.

### uvicorn host typo

- Use `--host 0.0.0.0` only if binding all interfaces intentionally; prod service uses **`127.0.0.1`** behind nginx.
- Wrong: `0.0..0` (IDNA error).

---

## Local developer setup (Linux / Mac)

Teammates clone the same repo; **no local Postgres required** if `.env` points at Azure.

```bash
git clone https://github.com/yasharthpaliwal-99/kellner-be.git
cd kellner-be
cp .env.example .env
# Fill PGSQL_*, Azure keys, Mongo, etc.

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

python scripts/setup_db.py
python scripts/embed_menu.py

python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### `ModuleNotFoundError: No module named 'fastapi'` but pip says “already installed”

- **Different Python** for `pip` vs `uvicorn`. Fix:

```bash
source .venv/bin/activate
which python   # must be .../kellner-be/.venv/bin/python
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### `setup_db.py` → socket `/var/run/postgresql/...`

- `PGSQL_ENDPOINT` empty in `.env` — psycopg2 falls back to local Postgres.
- Put `.env` in **repo root** (same folder as `app/`), not only under `scripts/`.

---

## Security checklist

- [ ] `.env` only on VM and dev machines; in `.gitignore`  
- [ ] Azure Postgres / Mongo firewall: VM IP + dev IPs  
- [ ] SSH key-only login; restrict port 22 where possible  
- [ ] Plan HTTPS (Let’s Encrypt) before wide production use  
- [ ] Rotate API keys if `.env` was ever shared in chat  

---

## Quick reference card

```bash
# Deploy
cd /opt/kellner && git pull origin yash-be-push && \
  source .venv/bin/activate && python -m pip install -r requirements.txt && \
  sudo systemctl restart kellner && curl -s http://127.0.0.1:8000/health

# Status
systemctl status kellner --no-pager
journalctl -u kellner -n 50 --no-pager
```

---

*Last updated for Kellner BE on Azure VM `Kellner-VM`, user `kellner-admin`, path `/opt/kellner`, branch `yash-be-push`.*
