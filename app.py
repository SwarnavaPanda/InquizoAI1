from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
import pdfplumber
from transformers import pipeline
import torch
import requests
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load summarization and QA pipelines
summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device=0 if torch.cuda.is_available() else -1)
qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2', device=0 if torch.cuda.is_available() else -1)

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'default_or_dev_value')
GOOGLE_CSE_ID = os.environ.get('GOOGLE_CSE_ID', 'default_or_dev_value')
SERPAPI_KEY = os.environ.get('SERPAPI_KEY', 'default_or_dev_value')

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_youtube(url):
    try:
        video_id = None
        if 'watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
        elif '/live/' in url:
            video_id = url.split('/live/')[1].split('?')[0]
        if video_id:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            return ' '.join([x['text'] for x in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        pass
    except Exception:
        pass
    return "No transcript or captions available for this video. Only videos with English transcripts or captions can be summarized."

def is_question_related_to_content(question, content):
    # Simple keyword overlap check (can be improved)
    question_words = set(question.lower().split())
    content_words = set(content.lower().split())
    overlap = question_words & content_words
    return len(overlap) > 0

def is_year_question(question):
    q = question.lower()
    return 'year' in q or 'when' in q

def answer_has_year(answer):
    return bool(re.search(r'\b(19|20)\d{2}\b', answer))

def ask_hf_llm(question, model_name):
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    prompt = f"Answer concisely: {question}"
    payload = {"inputs": prompt}
    try:
        resp = requests.post(api_url, json=payload, timeout=30)
        if resp.ok:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
                return data[0]['generated_text']
            elif isinstance(data, dict) and 'generated_text' in data:
                return data['generated_text']
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                return data[0]
    except Exception:
        pass
    return ""

def google_custom_search(query):
    if not GOOGLE_CSE_ID:
        return None
    url = (
        f"https://www.googleapis.com/customsearch/v1?q={query}"
        f"&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&num=1"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.ok:
            data = resp.json()
            if 'items' in data and len(data['items']) > 0:
                snippet = data['items'][0].get('snippet', '')
                return snippet
    except Exception:
        pass
    return None

def serpapi_search(query, is_year_question=False):
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.ok:
            data = resp.json()
            # Try to get direct answer from answer_box
            answer_box = data.get('answer_box', {})
            if is_year_question:
                # Check answer_box.answer
                if 'answer' in answer_box and answer_box['answer']:
                    year = extract_year(str(answer_box['answer']))
                    if year:
                        return year
                # Check answer_box.snippet
                if 'snippet' in answer_box and answer_box['snippet']:
                    year = extract_year(str(answer_box['snippet']))
                    if year:
                        return year
            else:
                if 'answer' in answer_box and answer_box['answer']:
                    return answer_box['answer']
                if 'snippet' in answer_box and answer_box['snippet']:
                    return answer_box['snippet']
            # Check knowledge_graph.description
            kg = data.get('knowledge_graph', {})
            if is_year_question and 'description' in kg and kg['description']:
                year = extract_year(str(kg['description']))
                if year:
                    return year
            elif 'description' in kg and kg['description']:
                return kg['description']
            # Check organic_results.snippet
            if 'organic_results' in data:
                for result in data['organic_results']:
                    snippet = result.get('snippet', '')
                    if is_year_question:
                        year = extract_year(snippet)
                        if year:
                            return year
                    elif snippet:
                        return snippet
    except Exception:
        pass
    return None

def extract_year(text):
    match = re.search(r'\b(19|20)\d{2}\b', text)
    if match:
        return match.group(0)
    return None

def is_definition_question(question):
    q = question.lower().strip()
    return q.startswith('what is') or q.startswith('define') or q.startswith('explain')

def is_generic_answer(answer):
    generic_phrases = [
        'how the brain works', 'no answer', 'not found', 'n/a', '', 'none', 'unknown', 'not provided', 'not available'
    ]
    return (
        len(answer.split()) < 4 or
        any(phrase in answer.lower() for phrase in generic_phrases)
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    data = request.form
    file = request.files.get('file')
    youtube_url = data.get('youtube_url')
    text_input = data.get('text_input')
    extracted_texts = []
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            file.save(tmp.name)
            extracted_texts.append(extract_text_from_pdf(tmp.name))
        os.unlink(tmp.name)
    if youtube_url:
        extracted_texts.append(extract_text_from_youtube(youtube_url))
    if text_input:
        extracted_texts.append(text_input)
    if not extracted_texts:
        return jsonify({'error': 'No valid input provided.'}), 400
    combined_text = '\n\n'.join([t for t in extracted_texts if t and t.strip()])
    return jsonify({'text': combined_text})

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided.'}), 400
    # Truncate text to 1500 characters to avoid model input limits
    truncated_text = text[:1500]
    try:
        summary = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': f'Summarization error: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    text = data.get('text')
    question = data.get('question')
    if not text or not question:
        return jsonify({'error': 'Text and question required.'}), 400
    sources = [s.strip() for s in text.split('\n\n') if s.strip()]
    best_answer = None
    best_score = 0
    for source in sources:
        qa_result = qa_pipeline({'context': source, 'question': question})
        answer = qa_result.get('answer', '').strip()
        score = qa_result.get('score', 0)
        if answer.lower() not in ["", "no answer", "not found", "n/a"] and score > best_score:
            best_answer = answer
            best_score = score
    # If a good answer is found from any source, return it (unless it's a definition or generic)
    if best_answer and best_score > 0.2 and not (is_definition_question(question) or is_generic_answer(best_answer)):
        return jsonify({'answer': best_answer})
    # If definition question or generic answer, use LLM
    llm_answer = ask_hf_llm(f"Question: {question}", "google/flan-t5-xl")
    if llm_answer and not is_generic_answer(llm_answer):
        return jsonify({'answer': llm_answer.strip()})
    # Fallback: check if question is related to any content
    if not any(is_question_related_to_content(question, s) for s in sources):
        return jsonify({'answer': 'Your question is not related with the given content'}), 200
    # Fallback to web/LLM as before
    try:
        search_url = f"https://api.duckduckgo.com/?q={question}&format=json&no_redirect=1&no_html=1"
        resp = requests.get(search_url)
        if resp.ok:
            data = resp.json()
            web_answer = data.get('AbstractText') or data.get('Answer') or data.get('RelatedTopics', [{}])[0].get('Text', '')
            if web_answer:
                return jsonify({'answer': web_answer})
    except Exception:
        pass
    return jsonify({'answer': 'Sufficient information is not provided.'})

@app.route('/internet_search')
def internet_search():
    question = request.args.get('q')
    if not question:
        return jsonify({'error': 'No question provided.'}), 400
    is_year_question = 'year' in question.lower() or 'when' in question.lower()
    # Try SerpAPI first
    serp_answer = serpapi_search(question, is_year_question)
    if serp_answer:
        return jsonify({'answer': serp_answer})
    # Try Google Custom Search next
    google_answer = google_custom_search(question)
    if google_answer:
        if is_year_question:
            year = extract_year(google_answer)
            if year:
                return jsonify({'answer': year})
        qa_result = qa_pipeline({"question": question, "context": google_answer})
        if qa_result and qa_result.get('score', 0) > 0.3 and qa_result.get('answer', '').strip().lower() not in ["", "no answer", "n/a"]:
            if is_year_question:
                year = extract_year(qa_result['answer'])
                if year:
                    return jsonify({'answer': year})
            return jsonify({'answer': qa_result['answer'].strip()})
        for model in ["google/flan-t5-xl", "bigscience/bloomz-560m"]:
            llm_answer = ask_hf_llm(f"Context: {google_answer}\nQuestion: {question}", model)
            if llm_answer and 'no answer' not in llm_answer.lower():
                if is_year_question:
                    year = extract_year(llm_answer)
                    if year:
                        return jsonify({'answer': year})
                return jsonify({'answer': llm_answer.strip()})
        return jsonify({'answer': google_answer})
    # Fallback to DuckDuckGo
    try:
        search_url = f"https://api.duckduckgo.com/?q={question}&format=json&no_redirect=1&no_html=1"
        resp = requests.get(search_url)
        if resp.ok:
            data = resp.json()
            web_answer = data.get('AbstractText') or data.get('Answer') or data.get('RelatedTopics', [{}])[0].get('Text', '')
            if web_answer:
                if is_year_question:
                    year = extract_year(web_answer)
                    if year:
                        return jsonify({'answer': year})
                qa_result = qa_pipeline({"question": question, "context": web_answer})
                if qa_result and qa_result.get('score', 0) > 0.3 and qa_result.get('answer', '').strip().lower() not in ["", "no answer", "n/a"]:
                    if is_year_question:
                        year = extract_year(qa_result['answer'])
                        if year:
                            return jsonify({'answer': year})
                    return jsonify({'answer': qa_result['answer'].strip()})
                for model in ["google/flan-t5-xl", "bigscience/bloomz-560m"]:
                    llm_answer = ask_hf_llm(f"Context: {web_answer}\nQuestion: {question}", model)
                    if llm_answer and 'no answer' not in llm_answer.lower():
                        if is_year_question:
                            year = extract_year(llm_answer)
                            if year:
                                return jsonify({'answer': year})
                        return jsonify({'answer': llm_answer.strip()})
                return jsonify({'answer': web_answer})
    except Exception:
        pass
    # Fallback to Hugging Face LLMs with explicit prompt
    for model in ["google/flan-t5-xl", "bigscience/bloomz-560m"]:
        llm_answer = ask_hf_llm(question, model)
        if llm_answer and 'no answer' not in llm_answer.lower():
            if is_year_question:
                year = extract_year(llm_answer)
                if year:
                    return jsonify({'answer': year})
            return jsonify({'answer': llm_answer.strip()})
    if is_year_question:
        return jsonify({'answer': 'Year not found in available sources.'})
    return jsonify({'answer': 'No answer found from LLM or internet. Please try rephrasing your question or check your internet connection.'})

@app.route('/healthz')
def healthz():
    return 'ok', 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 
