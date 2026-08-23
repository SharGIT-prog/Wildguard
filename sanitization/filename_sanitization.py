"""
WildGuard - sanitization/filename_sanitization.py

Always renames to a safe, deterministic filename that carries zero
location/context information - regardless of whether the original filename
was flagged as leaky (defense-in-depth, per spec section 23).
"""

import datetime as dt
import uuid


def generate_safe_filename(ext: str = "jpg") -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    clean_ext = ext.lower().lstrip(".")
    return f"sanitized_{stamp}_{short_id}.{clean_ext}"
