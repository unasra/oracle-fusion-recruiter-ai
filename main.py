from fastapi import FastAPI, Query , Request
from fastapi.middleware.cors import CORSMiddleware
#from fastapi import APIRouter, UploadFile, File, Form
from starlette.datastructures import UploadFile
from fastapi.responses import JSONResponse
import subprocess
from typing import List
import os
import json


app = FastAPI()

# Allow requests from your OracleCloud domain (or all origins for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://efpv.fa.us6.oraclecloud.com","https://fa-efpv-dev9-saasfaprod1.fa.ocs.oraclecloud.com"],  # Or ["*"] for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
async def search(request: Request):
    form = await request.form()

    query = form.get("query", None)


    # Collect files with keys like pdf_0, pdf_1, etc.
    pdf_files = []
    for key, value in form.items():
        print(type(value))
        if key.startswith("pdf") and isinstance(value, UploadFile):
            print(value)
            pdf_files.append(value)
    print(f"Received {len(pdf_files)} PDF files")

    os.makedirs("data", exist_ok=True)
    for file in os.listdir("data"):
        file_path = os.path.join("data", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save files into ./data folder
    saved_files = []
    for file in pdf_files:
        filename = file.filename
        save_path = os.path.join("data", filename)
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        saved_files.append(filename)

    # Run ai_Search.py with --reset flag using python3
    try:
        print("trying to run ai_Search.py with --reset flag")
        subprocess.run(["python3", "ai_populate_db.py","--reset"], check=True)
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": f"ai_Search.py failed: {str(e)}"})

    # Directly call query_rag_all_resumes with the query
    try:
        result = subprocess.run(["python3", "ai_search_query.py",f"{query}"], check=True,capture_output=True, text=True)
        scored_resumes = json.loads(result.stdout)
        print(scored_resumes)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"ai_search_query failed: {str(e)}"})

    return {"result": scored_resumes}

@app.post("/evaluate")
async def search(request: Request):
    form = await request.form()
    query = form.get("query", None)

    # Collect files with keys like pdf_0, pdf_1, etc.
    pdf_files = []
    for key, value in form.items():
        print(type(value))
        if key.startswith("pdf") and isinstance(value, UploadFile):
            print(value)
            pdf_files.append(value)
    print(f"Received {len(pdf_files)} PDF files")

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    for file in os.listdir("data"):
        file_path = os.path.join("data", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save files into ./data folder
    saved_files = []
    for file in pdf_files:
        filename = file.filename
        save_path = os.path.join("data", filename)
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        saved_files.append(filename)

    # Run ai_Search.py with --reset flag using python3
    try:
        print("trying to run ai_Search.py with --reset flag")
        subprocess.run(["python3", "ai_populate_db.py","--reset"], check=True)
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": f"ai_Search.py failed: {str(e)}"})

    # Directly call query_rag_evaluator with the query
    try:
        print("trying to run evaluate_data.py with query")
        result = subprocess.run(["python3", "evaluate_data.py",f"{query}"], check=True,capture_output=True, text=True)
        scored_resumes = json.loads(result.stdout)
        print(scored_resumes)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"ai_search_query failed: {str(e)}"})

    return {"result": scored_resumes}

@app.post("/linkedin")
async def linkedin(request: Request):
    try:
        data = await request.json()
        query = data.get("query", None)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {str(e)}"})
    print(query)

    # Run ai_Search.py with --reset flag using python3
    try:
        print("trying to run ai_Search.py with --reset flag")
        subprocess.run(["python3", "linkedin_evaluate.py","--reset"], check=True)
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": f"Linkedin Evaluation failed: {str(e)}"})

    # Directly call query_rag_all_resumes with the query
    try:
        result = subprocess.run(["python3", "ai_search_query.py", query], check=True, capture_output=True, text=True)
        scored_resumes = json.loads(result.stdout)
        print(scored_resumes[0])
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Linkedin Evaluation: {str(e)}"})

    return {"result": scored_resumes[0]}
