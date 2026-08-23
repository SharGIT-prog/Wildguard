"""
WildGuard - run_dashboard.py

Convenience entrypoint at the project root:
    python run_dashboard.py
is equivalent to:
    python -m dashboard.app
"""
import os
from dashboard.app import app
from config.settings import PROJECT_ROOT, MODEL_PATH

if __name__ == "__main__":
    print("WildGuard dashboard starting at http://127.0.0.1:5000")
    print(f"[WildGuard] Project root:  {os.path.abspath(PROJECT_ROOT)}")
    print(f"[WildGuard] Model path:    {os.path.abspath(MODEL_PATH)}  "
          f"({'FOUND' if os.path.isfile(MODEL_PATH) else 'NOT FOUND'})")
    app.run(host="127.0.0.1", port=5000, debug=False)
