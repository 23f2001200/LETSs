from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import sys
from io import StringIO
import traceback
import re

# Import Gemini
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class CodeRequest(BaseModel):
    code: str

# Response model
class CodeResponse(BaseModel):
    error: List[int]
    result: str

# Pydantic model for AI error analysis
class ErrorAnalysis(BaseModel):
    error_lines: List[int]

# Initialize Gemini client lazily
gemini_client = None

def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
        gemini_client = genai.Client(api_key=api_key)
    return gemini_client

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.
    Returns:
        {
            "success": bool,
            "output": str  # Exact stdout or traceback
        }
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Execute code
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception as e:
        # Get full traceback
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout

def extract_line_numbers_from_traceback(error_traceback: str) -> List[int]:
    """
    Extract line numbers from Python traceback.
    Looks for patterns like 'File "<string>", line X'.
    """
    line_numbers = []
    # Pattern to match line numbers in traceback from exec()
    pattern = r'File "<string>", line (\d+)'
    matches = re.findall(pattern, error_traceback)
    for match in matches:
        line_numbers.append(int(match))
    return line_numbers

def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """
    Use LLM with structured output to identify error line numbers.
    Falls back to regex extraction if AI fails.
    """
    try:
        client = get_gemini_client()

        prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{error_traceback}

Return the line number(s) where the error is located."""

        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "error_lines": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.INTEGER)
                        )
                    },
                    required=["error_lines"]
                )
            )
        )

        result = ErrorAnalysis.model_validate_json(response.text)
        return result.error_lines
    except Exception:
        # Fallback: extract line numbers from traceback using regex
        return extract_line_numbers_from_traceback(error_traceback)

@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):
    if not request.code or not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    
    # Execute the code
    execution_result = execute_python_code(request.code)
    
    if execution_result["success"]:
        # No errors
        return CodeResponse(error=[], result=execution_result["output"])
    else:
        # Error occurred - analyze to get line numbers
        error_lines = analyze_error_with_ai(request.code, execution_result["output"])
        return CodeResponse(error=error_lines, result=execution_result["output"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
