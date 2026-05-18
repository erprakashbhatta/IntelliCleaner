#!/usr/bin/env python3
import sys
import traceback

try:
    import server
    print("✓ Server imported successfully")
    print(f"✓ App has {len([r for r in server.app.url_map.iter_rules()])} routes")
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
