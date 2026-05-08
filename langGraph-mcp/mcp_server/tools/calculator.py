def calculate(expression: str) -> dict:
    """Safely evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return {"error": "Invalid characters in expression"}
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}