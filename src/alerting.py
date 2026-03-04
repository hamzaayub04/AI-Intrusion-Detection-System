import json
from datetime import datetime

LOG_FILE = "logs/alerts.json"

def log_alert(attack, confidence, features):
    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "attack": attack,
        "confidence": round(confidence, 3),
        "features": features
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(alert) + "\n")