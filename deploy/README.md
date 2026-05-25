# Production deploy (VM)

## Why prod dies when SSH closes

If you start the app with `uvicorn ...` in an SSH terminal, it is a **foreground child of that shell**. When you disconnect, the shell sends **SIGHUP** and the process exits.

Use **systemd** so the API runs as a system service and survives logout/reboot.

## One-time setup on the VM

```bash
cd /opt/kellner
git pull origin yash-be-push   # or your remote/branch
source .venv/bin/activate
pip install -r requirements.txt gunicorn

# Stop any manual uvicorn still running in another SSH session (Ctrl+C there).

sudo cp /opt/kellner/deploy/kellner.service /etc/systemd/system/kellner.service
sudo systemctl daemon-reload
sudo systemctl enable kellner
sudo systemctl start kellner
sudo systemctl status kellner
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

## After each code deploy

```bash
cd /opt/kellner
git pull origin yash-be-push
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart kellner
```

## Logs

```bash
journalctl -u kellner -f
```

## Nginx

If nginx proxies to the app, upstream should point at `http://127.0.0.1:8000` and include WebSocket upgrade headers for `/api/ws/conversation`.
