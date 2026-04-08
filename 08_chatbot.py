# ============================================================
# HealthBot-CRS | Step 8: Interactive Chatbot (CLI + Flask API)
# Uses: TF-IDF + SBERT + best classifier + hybrid recommender
# ============================================================

import pandas as pd
import numpy as np
import os, re, joblib, warnings, textwrap
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("=" * 60)
print(" HealthBot-CRS | Loading models...")
print("=" * 60)

# ============================================================
# LOAD ALL ARTIFACTS
# ============================================================
df = pd.read_csv("data/healthbot_clustered.csv")
embeddings = np.load("outputs/results/sbert_embeddings.npy")
emb_norm = normalize(embeddings)

le = LabelEncoder()
df["intent_encoded"] = le.fit_transform(df["intent_category"])
intent_names = list(le.classes_)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
    min_df=2, max_df=0.95, stop_words="english", sublinear_tf=True)
tfidf.fit(df["input_clean"])

sbert = SentenceTransformer("all-MiniLM-L6-v2")

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

X_full = np.hstack([tfidf.transform(df["input_clean"]).toarray(), embeddings])
y_full = df["intent_encoded"].values
_, X_test, _, y_test = train_test_split(X_full, y_full,
    test_size=0.2, random_state=42, stratify=y_full)

best_clf_name, best_clf, best_f1 = None, None, -1
for name in ["Logistic_Regression","Random_Forest","SVM","Gradient_Boosting","MLP"]:
    path = f"outputs/models/{name}.pkl"
    if os.path.exists(path):
        m = joblib.load(path)
        for attr in ['multi_class', 'l1_ratio']:
            if hasattr(m, attr):
                try:
                    delattr(m, attr)
                except:
                    pass

cf_model_name = "none"
for name in ["svd_recommender","nmf_recommender","knn_recommender"]:
    path = f"outputs/models/{name}.pkl"
    if os.path.exists(path):
        cf_model_raw = joblib.load(path)
        cf_model_name = name
        break

df["rating"] = (
    (df["entity_count"] / (df["entity_count"].max() + 1e-9)) * 2.5 +
    (df["output_length"] / (df["output_length"].max() + 1e-9)) * 2.5
).clip(0.5, 5.0)
n_intents = len(intent_names)
n_users = df["kmeans_sbert"].nunique()
R = np.zeros((n_users, n_intents))
for _, row in df.groupby(["kmeans_sbert","intent_encoded"])["rating"].mean().reset_index().iterrows():
    R[int(row["kmeans_sbert"]), int(row["intent_encoded"])] = row["rating"]

if "svd" in cf_model_name:
    U = cf_model_raw.transform(R)
    R_cf = np.dot(U, cf_model_raw.components_)
elif "nmf" in cf_model_name:
    W = cf_model_raw.transform(R)
    R_cf = np.dot(W, cf_model_raw.components_)
else:
    R_cf = R.copy()
R_cf = np.clip(R_cf, 0.5, 5.0)

print(f" Classifier : {best_clf_name} (F1={best_f1:.4f})")
print(f" CF model   : {cf_model_name}")
print(f" Dataset    : {len(df):,} responses | {n_intents} intents")
print(f" Ready!\n")

# ============================================================
# NLU LOOKUP TABLES
# ============================================================
GREETINGS = {"hi","hello","hey","hii","helo","howdy","good morning",
             "good evening","good afternoon","greetings","hi there","hello there"}

FAREWELLS = {"bye","goodbye","thank you","thanks","see you","cya",
             "thank you so much","thanks a lot","farewell","take care"}

INTENT_OVERRIDE = {
    "diabetes":"lifestyle","hypertension":"lifestyle",
    "blood pressure":"lifestyle","cholesterol":"lifestyle",
    "obesity":"lifestyle","diet":"lifestyle",
    "cancer":"diagnostics","tumor":"diagnostics","infection":"diagnostics",
    "fever":"symptom_inquiry","headache":"symptom_inquiry",
    "nausea":"symptom_inquiry","fatigue":"symptom_inquiry","rash":"symptom_inquiry",
    "ibuprofen":"medication_inquiry","antibiotic":"medication_inquiry",
    "paracetamol":"medication_inquiry","aspirin":"medication_inquiry",
    "anxiety":"mental_health","depression":"mental_health",
    "stress":"mental_health","sleep":"mental_health",
    "chest pain":"emergency","heart attack":"emergency",
    "stroke":"emergency","cant breathe":"emergency",
    "shortness of breath":"emergency",
}

FOLLOWUPS = {
    "lifestyle":         ["What foods should I avoid?","How much exercise is recommended?","Can lifestyle changes reverse this?","What are the warning signs?"],
    "symptom_inquiry":   ["How long do these symptoms last?","When should I see a doctor?","What medication helps?","Could this be serious?"],
    "medication_inquiry":["What is the correct dosage?","Are there side effects?","Can I take this with other medications?","What are the alternatives?"],
    "mental_health":     ["How do I manage this daily?","Are there breathing techniques?","When should I seek professional help?","How does sleep affect this?"],
    "diagnostics":       ["What tests are needed?","How is this diagnosed?","What are the treatment options?","What are the stages?"],
    "emergency":         ["What are signs of worsening?","When should I call 911?","What to do while waiting for help?","What causes this condition?"],
    "greeting":          ["I have a headache and fever","Side effects of ibuprofen?","I feel anxious and cannot sleep","How to manage type 2 diabetes?"],
    "farewell":          [],
    "unclear":           ["Can you rephrase your question?","Tell me your main symptom","Are you asking about a medication?","Do you need emergency help?"],
    "general_inquiry":   ["Tell me more about this","What lifestyle changes help?","Are there medications for this?","When should I see a doctor?"],
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def clean_text(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()

def clean_response(text):
    text = re.sub(
        r"^(hi[.,]?\s*|hello[.,]?\s*|hey[.,]?\s*|"
        r"thanks?\s+for\s+(your\s+query|using\s+chat\s*doctor)[,.]?\s*|"
        r"thank\s+you\s+for\s+(your\s+query|using\s+chat\s*doctor)[,.]?\s*|"
        r"i\s+have\s+gone\s+through\s+(all\s+)?the\s+(data|information)\s+you\s+(have\s+)?posted[,.]?\s*)",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(r"chat\s*doctor", "HealthBot-CRS", text, flags=re.IGNORECASE)
    text = text.strip()
    return text[0].upper() + text[1:] if text else text

# ============================================================
# CORE RESPONSE FUNCTION
# ============================================================
def get_response(user_input, top_k=3, alpha=0.6, verbose=False):
    normalized = clean_text(user_input).strip()

    if normalized in GREETINGS:
        return {"intent":"greeting","confidence":1.0,
                "response":"Hello! I am HealthBot-CRS. I can help you with symptoms, medications, diagnostics, mental health, lifestyle and emergencies. What can I assist you with today?",
                "alternatives":[],"followups":FOLLOWUPS["greeting"],"user_cluster":0}

    if normalized in FAREWELLS:
        return {"intent":"farewell","confidence":1.0,
                "response":"Thank you for using HealthBot-CRS. Stay healthy and take care! Remember to consult a qualified healthcare professional for all medical decisions.",
                "alternatives":[],"followups":[],"user_cluster":0}

    cleaned = clean_text(user_input)

    tfidf_vec = tfidf.transform([cleaned]).toarray()
    sbert_vec = sbert.encode([cleaned], convert_to_numpy=True)
    X_q = np.hstack([tfidf_vec, sbert_vec])

    intent_idx = None
    confidence = None
    for kw, forced_intent in INTENT_OVERRIDE.items():
        if kw in cleaned and forced_intent in intent_names:
            intent_idx = intent_names.index(forced_intent)
            confidence = 0.92
            break
    if intent_idx is None:
        intent_idx = best_clf.predict(X_q)[0]
        confidence = float(best_clf.predict_proba(X_q).max())

    intent_name = intent_names[intent_idx]

    if confidence < 0.45:
        return {"intent":"unclear","confidence":confidence,
                "response":"I am not fully sure I understood that. Could you describe your symptoms or question in more detail? For example: \"I have a headache and fever\" or \"What are the side effects of ibuprofen?\"",
                "alternatives":[],"followups":FOLLOWUPS["unclear"],"user_cluster":0}

    sbert_q_norm = normalize(sbert_vec)
    cb_scores = cosine_similarity(sbert_q_norm, emb_norm)[0]

    intent_mask = df["intent_encoded"].values == intent_idx
    if intent_mask.sum() == 0:
        intent_mask = np.ones(len(df), dtype=bool)

    cluster_sbert = normalize(
        np.array([
            emb_norm[df["kmeans_sbert"] == c].mean(axis=0)
            if (df["kmeans_sbert"] == c).sum() > 0
            else np.zeros(emb_norm.shape[1])
            for c in range(n_users)
        ])
    )
    user_cluster = cosine_similarity(sbert_q_norm, cluster_sbert).argmax()
    cf_intent_scores = R_cf[user_cluster]
    cf_row = np.array([cf_intent_scores[df.iloc[i]["intent_encoded"]]
                       for i in range(len(df))])
    cf_norm = (cf_row - cf_row.min()) / (cf_row.max() - cf_row.min() + 1e-9)

    hybrid = alpha * cb_scores + (1 - alpha) * cf_norm
    hybrid[~intent_mask] = -1

    top_idx = np.argsort(hybrid)[::-1][:top_k]
    top_responses = [clean_response(df.iloc[i]["output"]) for i in top_idx]

    return {
        "intent":       intent_name,
        "confidence":   confidence,
        "response":     top_responses[0],
        "alternatives": top_responses[1:],
        "followups":    FOLLOWUPS.get(intent_name, FOLLOWUPS["general_inquiry"]),
        "user_cluster": int(user_cluster),
    }

# ============================================================
# CLI CHATBOT
# ============================================================
BANNER = """
╔══════════════════════════════════════════════════╗
║  HealthBot-CRS                                   ║
║  Clinical Response System | Type quit to exit    ║
╚══════════════════════════════════════════════════╝
"""
DISCLAIMER = (
    "DISCLAIMER: HealthBot-CRS is a research prototype.\n"
    " It does NOT provide medical advice. Always consult a\n"
    " qualified healthcare professional for medical decisions.\n"
)

def run_cli():
    print(BANNER)
    print(DISCLAIMER)
    print("-" * 52)
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit","exit","q"}:
            print("\nGoodbye! Stay healthy.")
            break
        result = get_response(user_input)
        print(f"\nHealthBot [{result['intent']} | {result['confidence']*100:.0f}%]:")
        for line in textwrap.wrap(result["response"], width=70):
            print(f"  {line}")
        if result.get("followups"):
            print("\n  Follow-up questions:")
            for q in result["followups"]:
                print(f"   > {q}")
        print("\n" + "-"*52)

# ============================================================
# FLASK API
# ============================================================
def run_api(port=7860):
    try:
        from flask import Flask, request, jsonify, send_from_directory
        from flask_cors import CORS
    except ImportError:
        print("Run: pip install flask flask-cors")
        return

    app = Flask(__name__, static_folder=".")
    CORS(app)

    @app.route("/")
    def index():
        base = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(base, "HealthBot_CRS.html")):
            return send_from_directory(base, "HealthBot_CRS.html")
        return "HealthBot_CRS_chat.html not found in " + base, 404

    @app.route("/chatui") 
    @app.route("/HealthBot_CRS_Chat.html")
    def chat_page():
        base = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(base, "HealthBot_CRS_Chat.html")):
            return send_from_directory(base, "HealthBot_CRS_Chat.html")
        return "Not found", 404

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status":"ok","model":best_clf_name})

    @app.route("/chat", methods=["POST"])
    def chat():
        import traceback
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400
        try:
            result = get_response(data["message"],
                top_k=int(data.get("top_k", 3)),
                alpha=float(data.get("alpha", 0.6)))
            return jsonify({
                "intent":       result["intent"],
                "confidence":   round(result["confidence"], 4),
                "response":     result["response"],
                "alternatives": result["alternatives"],
                "followups":    result.get("followups", []),
            })
        except Exception as e:
            print("❌ CHAT ERROR:", traceback.format_exc())
            return jsonify({
                "intent": "error",
                "confidence": 0.0,
                "response": str(e),
                "alternatives": [],
                "followups": []
            }), 200

    @app.route("/intents", methods=["GET"])
    def intents():
        return jsonify({"intents":intent_names})

    print(f" HealthBot running!")
    print(f" Open: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys
    if "--api" in sys.argv:
        idx = sys.argv.index("--api")
        port = int(sys.argv[idx+1]) if len(sys.argv) > idx+1 else 7860
        run_api(port)
    else:
        run_cli()
