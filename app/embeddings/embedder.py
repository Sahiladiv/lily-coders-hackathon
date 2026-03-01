# Local embedding model

import mysql.connector
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import uuid

# --------------------------------------------
# LM STUDIO CLIENT
# --------------------------------------------

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # can be anything
)

EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"

# --------------------------------------------
# MYSQL CONFIG
# --------------------------------------------

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "safety_monitoring"
}

# --------------------------------------------
# CONNECT TO MYSQL
# --------------------------------------------

conn = mysql.connector.connect(**MYSQL_CONFIG)
cursor = conn.cursor(dictionary=True)

cursor.execute("SELECT * FROM incidents")
rows = cursor.fetchall()

print(f"Loaded {len(rows)} incidents from MySQL")

# --------------------------------------------
# FORMAT INCIDENT FOR RAG
# --------------------------------------------

def format_incident(row):
    return f"""
Incident ID: {row['id']}
Image: {row['image_path']}
Severity: {row['severity']}
Decision: {row['final_decision']}
Compliance Score: {row['compliance_score']}
Detected PPE: {row['detected_ppe']}
Missing PPE: {row['missing_ppe']}
OSHA Codes: {row['osha_codes']}
Escalation: {row['escalation']}
Report: {row['report']}
Timestamp: {row['created_at']}
"""

documents = []
ids = []

for row in rows:
    documents.append(format_incident(row))
    ids.append(str(uuid.uuid4()))

# --------------------------------------------
# GENERATE EMBEDDINGS VIA LM STUDIO
# --------------------------------------------

response = client.embeddings.create(
    model=EMBED_MODEL,
    input=documents
)

embeddings = [item.embedding for item in response.data]

# --------------------------------------------
# INIT CHROMA
# --------------------------------------------

chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        is_persistent=True
    )
)

collection = chroma_client.get_or_create_collection(
    name="incident_reports"
)

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)

print("✅ Embedded into Chroma successfully!")

cursor.close()
conn.close()