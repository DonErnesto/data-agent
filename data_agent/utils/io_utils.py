def safe_load_json(path: str):
    import json

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}
