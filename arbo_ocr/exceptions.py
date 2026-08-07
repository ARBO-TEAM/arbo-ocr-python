class OcrError(RuntimeError):
    """Raised when the arboocr_demo process fails to start, exits non-zero,
    or produces stdout that isn't valid JSON. An empty `lines` list in a
    successful (exit-0) PageResult is NOT an error — arboOCR's own contract
    is that "no text found" is a normal, valid result."""

    def __init__(self, message: str, exit_code: int = -1, stderr: str = ""):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr
