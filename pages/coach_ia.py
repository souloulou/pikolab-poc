"""
PikoLab — Coach IA (page separee)
Chat conversationnel avec Iris, coach styliste virtuelle.
Lit le contexte d'analyse depuis st.session_state.
"""

import os
import streamlit as st
from google import genai

# Import shared functions
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import build_coach_system_prompt, GEMINI_MODELS

st.set_page_config(
    page_title="PikoLab — Coach Iris",
    page_icon="💬",
    layout="centered",
)

# Hide auto-generated page nav + custom sidebar
st.markdown(
    "<style>[data-testid='stSidebarNav'] { display: none !important; }</style>",
    unsafe_allow_html=True,
)
st.sidebar.header("PikoLab")
st.sidebar.page_link("app.py", label="Analyse", icon="🎨")
st.sidebar.page_link("pages/coach_ia.py", label="Coach Iris", icon="💬")

# ---- Check prerequisites ----
try:
    ai_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
except Exception:
    ai_key = os.environ.get("GEMINI_API_KEY", "")

if not ai_key:
    st.warning("Cle API Gemini non configuree. Ajoutez GEMINI_API_KEY dans .streamlit/secrets.toml")
    st.stop()

ctx = st.session_state.get("ctx")
if not ctx:
    st.info("Analysez d'abord une photo sur la page principale pour activer le coach.")
    if st.button("Aller a l'analyse", type="primary", use_container_width=True):
        st.switch_page("app.py")
    st.stop()

# ---- Header ----
season = ctx["season"]
advice = ctx["advice"]
profile = ctx["profile"]

st.markdown(f"### Coach Iris — {season}")
st.caption(
    "Posez vos questions : tenue pour une occasion, shopping, "
    "maquillage, cheveux... Iris connait votre profil complet."
)

# ---- Build system prompt ----
quiz_data = {}
if "q_hair" in st.session_state:
    quiz_data["hair_dyed"] = st.session_state.get("q_hair", "")
    quiz_data["natural_hair"] = st.session_state.get("q_nat_hair", "")
    quiz_data["style"] = st.session_state.get("q_style", "")
    quiz_data["work"] = st.session_state.get("q_work", "")
    quiz_data["current_colors"] = ", ".join(st.session_state.get("q_colors", []))
    quiz_data["interest"] = st.session_state.get("q_interest", "")

system_prompt = build_coach_system_prompt(
    season, advice, profile,
    ctx.get("diagnostic", []),
    ctx.get("hair_info", {}),
    ctx.get("lip_undertone", "inconnu"),
    quiz_data,
)

# ---- Chat ----
if "coach_messages" not in st.session_state:
    st.session_state.coach_messages = []

for msg in st.session_state.coach_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: Je vais a un mariage, qu'est-ce que je porte ?"):
    st.session_state.coach_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            client = genai.Client(api_key=ai_key)
            contents = []
            for m in st.session_state.coach_messages[:-1]:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
            contents.append({"role": "user", "parts": [{"text": prompt}]})

            response_text = ""
            last_error = None
            for model_name in GEMINI_MODELS:
                try:
                    stream = client.models.generate_content_stream(
                        model=model_name,
                        contents=contents,
                        config={"system_instruction": system_prompt},
                    )

                    def _stream():
                        for chunk in stream:
                            if chunk.text:
                                yield chunk.text

                    response_text = st.write_stream(_stream())
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if "429" not in str(exc) and "quota" not in str(exc).lower():
                        raise
                    continue

            if last_error:
                response_text = f"Quota Gemini epuisee. Reessayez dans quelques minutes."
                st.warning(response_text)

            st.session_state.coach_messages.append({"role": "assistant", "content": response_text})
        except Exception as exc:
            err_msg = f"Erreur : {exc}"
            st.error(err_msg)
            st.session_state.coach_messages.append({"role": "assistant", "content": err_msg})
