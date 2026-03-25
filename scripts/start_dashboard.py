"""Start the Kalshi trading dashboard on port 8050."""
import sys
sys.path.insert(0, ".")

import uvicorn

if __name__ == "__main__":
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8050, reload=False)
