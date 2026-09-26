# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ |
| 1.0.0-beta | ✅ (beta) |
| 0.1.x   | ⚠️ deprecated — upgrade to 1.0.0-beta |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly:

**Do NOT open a public GitHub issue.**

Instead, email us at: security@lm-optimizer.example.com (replace with actual contact)

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes

We will:
1. Acknowledge within 48 hours
2. Investigate and validate
3. Develop a fix
4. Release a patch
5. Credit you (if desired)

## Security Considerations

### LM Studio Connection

The application connects to your LM Studio instance via HTTP API. Security considerations:

- **Local by default**: Web UI binds to `127.0.0.1:8080` only
- **Remote LM Studio**: Can connect to remote instances (e.g., Tailscale, VPN)
- **No authentication**: LM Studio API typically doesn't require auth
- **Network exposure**: Ensure LM Studio port (1234) is not publicly exposed

### Data Storage

- **SQLite database**: Stored locally in `./data/optimizer.db`
- **No credentials stored**: LM Studio URL only, no API keys
- **Benchmark outputs**: Stored locally, may contain generated text
- **Presets**: Configuration only, no model weights

### Web UI

- **No authentication**: Designed for local/trusted network use
- **CORS**: Restricted to localhost by default
- **WebSocket**: Used for real-time updates
- **HTTPS**: Not configured by default (use reverse proxy for production)

### Best Practices

1. **Don't expose to public internet** without proper authentication
2. **Use VPN/Tailscale** for remote LM Studio access
3. **Keep LM Studio updated** to latest version
4. **Monitor logs** for unexpected behavior
5. **Backup database** before major operations

## Threat Model

| Threat | Likelihood | Impact | Mitigation |
|--------|------------|--------|------------|
| Unauthorized LM Studio access | Medium | High | Bind to localhost, use VPN |
| Data exfiltration | Low | Medium | Local storage only |
| Model tampering | Low | High | Read-only model access |
| DoS via optimization | Medium | Medium | Timeout limits, resource monitoring |

## Secure Deployment

For production/team use:

1. **Reverse proxy** with authentication (nginx + auth, Cloudflare Access, etc.)
2. **HTTPS** with valid certificates
3. **Firewall** restricting access to trusted IPs
4. **Regular updates** of dependencies
5. **Audit logs** for optimization runs

## Dependencies

We regularly update dependencies. Check `pyproject.toml` for current versions.

Run security audit:
```bash
pip-audit
```

## Contact

For security questions: security@lm-optimizer.example.com

For general support: GitHub Issues or Discussions