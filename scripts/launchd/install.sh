#!/usr/bin/env bash
# Install DriftEdge continuous polling daemon to launchd.
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$HOME/Library/LaunchAgents"
mkdir -p "$DST"

label="com.tanishk.driftedge.poll"
plist="${label}.plist"

if launchctl list | grep -q "$label"; then
    echo "Unloading existing $label..."
    launchctl unload "$DST/$plist" 2>/dev/null || true
fi

cp "$SRC/$plist" "$DST/$plist"
launchctl load "$DST/$plist"
echo "Loaded $label -> $DST/$plist"
echo
echo "Verify with:  launchctl list | grep driftedge"
echo "Tail logs:    tail -f logs/launchd-poll.out"
