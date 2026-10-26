"""Who is talking to LM Studio (:1234) right now?"""

import subprocess

out = subprocess.run(
    ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True
).stdout
pid_cmd = {}
for line in out.splitlines():
    if "127.0.0.1:1234" in line and ("ESTABLISHED" in line or "SYN_SENT" in line):
        cols = line.split()
        pid = cols[-1]
        proto = cols[0]
        local, foreign, state = cols[1], cols[2], cols[3]
        key = (proto, local, foreign, state, pid)
        print("  ", key)

pids = {c[-1] for c in []}
# collect owning pids again for naming
seen = set()
for line in out.splitlines():
    if "127.0.0.1:1234" in line and ("ESTABLISHED" in line or "SYN_SENT" in line):
        cols = line.split()
        pid = cols[-1]
        if pid in seen:
            continue
        seen.add(pid)
        try:
            info = subprocess.run(
                ["wmic", "process", "where", f"ProcessId={pid}",
                 "get", "ProcessId,Name,CommandLine", "/format:list"],
                capture_output=True, text=True,
            ).stdout.replace("\r", "")
            print(f"--- PID {pid} ---")
            for ln in info.splitlines():
                if ln.strip():
                    print("   ", ln.strip()[:200])
        except Exception as e:
            print("pid lookup failed", pid, e)
if not seen:
    print("no connections to :1234 at this moment")
