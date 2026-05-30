#!/usr/bin/env bash
set -euo pipefail
DST="$HOME/Library/LaunchAgents"
label="com.tanishk.driftedge.poll"
plist="$DST/${label}.plist"
if [[ -f "$plist" ]]; then
    launchctl unload "$plist" 2>/dev/null || true
    rm -f "$plist"
    echo "Removed $label"
fi
