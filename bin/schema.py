upload_request_schema = {
    "type": "object",
    "properties": {
        "context": {
            "type": "string"
        },
        "category": {
            "type": "string"
        },
        "threshold": {
            "type": "number"
        },
        "noOfMatches": {
            "type": "integer"
        },
        "inputPath": {
            "type": "string"
        }
    },
    "required": ["context", "category", "threshold", "noOfMatches", "inputPath"],
    "additionalProperties": False
}
