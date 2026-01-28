#!/bin/bash
# Hook: Enforce ClearML Cloud config for this project
# Ensures CLEARML_CONFIG_FILE points to cloud and stale credential env vars are cleared
# Exit code 2 = blocking error (prevents tool call)

# Read tool input from stdin to check if this is a clearml-related command
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

# Only check for clearml-related commands
case "$COMMAND" in
    *clearml*|*uv\ run\ dapidl*|*CLEARML*) ;;
    *) exit 0 ;;
esac

# Check for stale credential env vars that override config file
if [ -n "$CLEARML_API_ACCESS_KEY" ] && [ "$CLEARML_API_ACCESS_KEY" != "0G0GWAC1W2XMMU3RFQ3VC5Z4829S4B" ]; then
    echo "BLOCKED: CLEARML_API_ACCESS_KEY env var is set to non-cloud credentials. This project uses ClearML Cloud (app.clear.ml). Run: unset CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY" >&2
    exit 2
fi

# Warn (non-blocking) if config file not set
if [ -z "$CLEARML_CONFIG_FILE" ] || [ "$CLEARML_CONFIG_FILE" != "$HOME/.clearml/clearml-cloud.conf" ]; then
    echo "WARN: CLEARML_CONFIG_FILE not set to cloud config. Set with: export CLEARML_CONFIG_FILE=\$HOME/.clearml/clearml-cloud.conf" >&2
fi

exit 0
