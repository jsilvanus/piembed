# systemd user unit

`piembed.service` runs piembed as a systemd user service.

Expected layout (relative to the service user's home, `%h`):

- app checkout (built, with `dist/`): `~/apps/piembed`
- environment file: `~/apps/piembed.env` (all settings and secrets live here, never in git)
- Node.js: `~/.local/bin/node`

Install:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/piembed.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now piembed
loginctl enable-linger "$USER"   # once, so it runs without a login session
```

Status and logs: `systemctl --user status piembed`, `journalctl --user -u piembed -f`.
