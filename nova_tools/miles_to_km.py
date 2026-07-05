def tool_miles_to_km(miles: float) -> str:
    """Convert miles to kilometers."""
    try:
        km = float(miles) * 1.60934
        return f"{float(miles)} miles is {km:.2f} kilometers."
    except (TypeError, ValueError):
        return "miles_to_km error: please pass a number."

SPEC = {
    "type": "function",
    "function": {
        "name": "miles_to_km",
        "description": "Convert a distance in miles to kilometers.",
        "parameters": {
            "type": "object",
            "properties": {"miles": {"type": "number", "description": "distance in miles"}},
            "required": ["miles"],
        },
    },
}

if __name__ == "__main__":
    print(tool_miles_to_km(10))
    print(tool_miles_to_km("oops"))
