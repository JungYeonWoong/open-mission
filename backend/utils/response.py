from datetime import datetime
from typing import Any, Optional

def api_response(
    success: bool,
    message: str = "",
    data: Optional[Any] = None,
    error: Optional[str] = None,
):
    return {
        "success": success,
        "message": message,
        "error": error,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }
