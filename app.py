# app.py
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import traceback

# ---------------- Flask ----------------
app = Flask(__name__)

# ---------------- Env ----------------
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set. Add it to your environment or .env.")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env.")

# Expose for downstream libs
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
# Map for Google SDKs
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ------------- LangChain compat shim -------------
# Prevents AttributeError: langchain.debug / langchain.llm_cache, etc.
try:
    import langchain as _lc
    # Provide legacy flags
    if not hasattr(_lc, "verbose"):
        _lc.verbose = False
    if not hasattr(_lc, "debug"):
        _lc.debug = False
    if not hasattr(_lc, "tracing_v2_enabled"):
        _lc.tracing_v2_enabled = False
    # Provide legacy cache attribute expected by some code paths
    if not hasattr(_lc, "llm_cache"):
        _lc.llm_cache = None
    # Also set the new global cache implementation to None
    try:
        from langchain_core.globals import set_llm_cache
        set_llm_cache(None)
    except Exception:
        pass
except Exception:
    pass

# ------------- Imports after env ready -------------
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompt import *  # should define `system_prompt`

# ------------- Vector Store / Retriever -------------
embeddings = download_hugging_face_embeddings()
INDEX_NAME = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ------------- Gemini LLM -------------
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.4,
    max_output_tokens=500,
    # If needed in your install, uncomment next line:
    # google_api_key=os.environ["GOOGLE_API_KEY"],
)

# ------------- RAG chain -------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        # Accept msg from query, form, JSON, or raw text
        msg = None

        if not msg:
            msg = request.args.get("msg")

        if not msg and request.form:
            msg = request.form.get("msg")
            if not msg:
                for k in ("message", "text", "q", "query"):
                    if k in request.form:
                        msg = request.form[k]
                        break

        if not msg and request.is_json:
            data = request.get_json(silent=True) or {}
            for k in ("msg", "message", "text", "q", "query"):
                if k in data:
                    msg = data[k]
                    break

        if not msg:
            raw = (request.get_data(as_text=True) or "").strip()
            if raw:
                msg = raw

        if not msg or not str(msg).strip():
            print(
                "DEBUG /get payload -> args:", dict(request.args),
                " form:", dict(request.form) if request.form else {},
                " is_json:", request.is_json
            )
            return jsonify({
                "error": "Empty message payload",
                "hint": "Send 'msg' via form, JSON, query-string, or raw text."
            }), 400

        msg = str(msg).strip()

        result = rag_chain.invoke({"input": msg})
        answer = (
            result.get("answer")
            if isinstance(result, dict) else
            (str(result) if result is not None else "")
        )
        if not answer:
            answer = "Sorry, I couldn't find an answer for that."

        return jsonify({"answer": answer}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Unexpected error while answering.",
            "detail": str(e),
            "answer": ""
        }), 500

# --------------- Main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
