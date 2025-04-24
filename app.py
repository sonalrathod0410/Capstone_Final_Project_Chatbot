# --- IMPORTS ---
import chainlit as cl
import ollama
import PyPDF2
import re
import time
import asyncio
import docx
from fuzzywuzzy import fuzz
import spacy
import numpy as np

# --- NLP MODEL ---
nlp = spacy.load("en_core_web_sm")

# --- PDF PATH ---
pdf_path = r"C:\\Sonal\\Sem4\\Cap\\Project\\local-chatgpt with Gemma 3\\ai-engineering-hub\\local-chatgpt with Gemma 3\\2025-26-Handbook.pdf"

# --- SUPPORT SERVICE LINKS ---
support_keywords = {
    "scholarship": "https://www.stclaircollege.ca/foundation/scholarships",
    "bursary": "https://www.stclaircollege.ca/foundation/scholarships",
    "financial aid": "https://www.stclaircollege.ca/financial-aid",
    "counselling": "https://www.stclaircollege.ca/student-services/counselling",
    "counseling": "https://www.stclaircollege.ca/student-services/counselling",
    "tutoring": "https://www.stclaircollege.ca/student-services/tutoring",
    "career fair": "https://www.stclaircollege.ca/career-services",
    "svp": "https://www.stclaircollege.ca/svp",
    "apply": "https://www.stclaircollege.ca/future-students/apply",
    "health insurance": "https://www.stclaircollege.ca/health-centre/fees",
    "registrar": "https://www.stclaircollege.ca/registrars-office",
    "faq": "https://www.stclaircollege.ca/registrars-office/faq",
    "benefit": "https://www.stclaircollege.ca/student-life",
    "IT services": "https://www.stclaircollege.ca/it-services",
    "Library services": "https://www.stclaircollege.ca/library",
    "Tuition fees": "https://www.stclaircollege.ca/international/tuition-fees",
    "fees": "https://www.stclaircollege.ca/international/tuition-fees",
    "Contact us": "https://www.stclaircollege.ca/international/contact",
    "pgwp-eligible-programs": "https://www.stclaircollege.ca/international/pgwp-eligible-programs",
    "Pre-Departure Checklist": "https://www.stclaircollege.ca/international/arrival-information",
    "Visa information": "https://www.stclaircollege.ca/international/visa",
    "withdrawal-refund-policy": "https://www.stclaircollege.ca/international/withdrawal-refund-policy",
    "work opportunities": "https://www.stclaircollege.ca/international/work",
    "Housing for international students": "https://www.stclaircollege.ca/international/housing",
    "How to make payment": "https://www.stclaircollege.ca/international/how-to-make-a-payment",
    "Admission Process": "https://www.stclaircollege.ca/programs/admission-procedures",
    "food services": "https://www.stclaircollege.ca/student-services/on-campus-services/food-services"
}

# --- SUPPORT LINK MATCHING ---
def match_support_service(query, threshold=70):
    best_match = None
    best_score = 0
    for keyword, url in support_keywords.items():
        score = fuzz.partial_ratio(query.lower(), keyword.lower())
        if score > best_score and score >= threshold:
            best_score = score
            best_match = (keyword, url)
    return best_match

# --- PDF TEXT EXTRACTION HELPERS ---
def extract_field(pattern, text, flags=0):
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else "Not specified"

def extract_all_campuses(text):
    return re.findall(r"(Downtown Campus|Chatham Campus|Main Windsor Campus):\s*([A-Z0-9]+)", text)

def extract_course_description(text):
    desc_match = re.search(r"(?:DESCRIPTION|Program Overview|PROGRAM OVERVIEW)(.*?)(?=CAREER OPPORTUNITIES|ADMISSION REQUIREMENTS|Start Date|$)", text, flags=re.DOTALL | re.IGNORECASE)
    return desc_match.group(1).strip() if desc_match else "Not specified"

def extract_duration(text):
    match = re.search(r"(One Year|Two Year|Three Year|Four Year).*?Start Date", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).title()
    fallback = re.search(r"(One Year|Two Year|Three Year|Four Year)", text, re.IGNORECASE)
    return fallback.group(1).title() if fallback else "Not specified"

def infer_duration_by_similarity(title, campus, code, full_text):
    candidates = re.findall(rf"({title.upper()}.*?(?:Downtown Campus|Chatham Campus|Main Windsor Campus):\s*{code}.*?)(?=\n[A-Z][A-Z\s\-/&]+\n|$)", full_text, re.DOTALL)
    for block in candidates:
        duration = extract_duration(block)
        if duration and duration.lower() not in ["not specified", "two year"]:
            return duration
    return "Not specified"

def split_program_entries(full_text):
    pattern = r"\n(?=[A-Z][A-Z\s\-/&]+(?: FOR [A-Z\s]+)?\n(?:Downtown Campus|Chatham Campus|Main Windsor Campus):)"
    return [entry.strip() for entry in re.split(pattern, full_text) if entry.strip()]

def fetch_programs_from_pdf(pdf_path):
    programs = []
    seen_titles = set()
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            raw_entries = split_program_entries(full_text)

            for entry in raw_entries:
                lines = entry.strip().splitlines()
                if not lines:
                    continue
                title = lines[0].strip()
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                body = "\n".join(lines[1:])
                campus_codes = extract_all_campuses(body)
                start_date = extract_field(r"Starts:\s*(\w+)", body)
                career_opportunities = extract_field(r"CAREER OPPORTUNITIES(.*?)(?=ADMISSION REQUIREMENTS|$)", body, flags=re.DOTALL)
                admission_requirements = extract_field(r"ADMISSION REQUIREMENTS(.*?)(?:Check|$)", body, flags=re.DOTALL)
                course_description = extract_course_description(body)
                if course_description == "Not specified":
                    course_description = f"Admission Requirements: {admission_requirements}"

                for campus_name, course_code in campus_codes:
                    single_block = entry
                    duration = extract_duration(single_block)
                    if not duration or duration.lower() == "not specified":
                        duration = infer_duration_by_similarity(title, campus_name, course_code, full_text)
                    if not duration or duration.lower() == "not specified":
                        duration = "Duration information not available"

                    programs.append({
                        "title": title,
                        "campus_codes": [{"campus": campus_name, "code": course_code}],
                        "start_date": start_date,
                        "duration": duration,
                        "career_opportunities": career_opportunities,
                        "admission_requirements": admission_requirements,
                        "course_description": course_description,
                        "course_code": course_code
                    })
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return programs

# --- RESUME TEXT PARSING ---
def parse_resume_text(file_path):
    if file_path.endswith(".pdf"):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    return ""

# --- SEMANTIC MATCHING (SPAcy VECTOR COSINE SIMILARITY) ---
def semantic_program_match(resume_text, programs, threshold=0.75):
    doc_resume = nlp(resume_text)
    results = []
    for program in programs:
        desc = program.get("course_description", "")
        if not desc.strip():
            continue
        doc_course = nlp(desc)
        if doc_resume.vector_norm and doc_course.vector_norm:
            sim = doc_resume.similarity(doc_course)
            if sim >= threshold:
                results.append((program, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]

# --- CHATBOT HANDLERS ---
@cl.on_chat_start
async def start_chat():
    if not cl.user_session.get("interaction"):
        cl.user_session.set("interaction", [{"role": "system", "content": "You are a helpful assistant."}])
    await cl.Message(content="👋 Welcome to St. Clair Chatbot! Upload your resume or ask about a course.").send()

    loop = asyncio.get_event_loop()
    programs_pdf = await loop.run_in_executor(None, fetch_programs_from_pdf, pdf_path)
    cl.user_session.set("programs", programs_pdf)

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        if message.elements:
            file = message.elements[0]
            resume_text = parse_resume_text(file.path)
            cl.user_session.set("resume_text", resume_text)
            await cl.Message(content="📄 Resume uploaded and analyzed successfully. You can now ask for course suggestions.").send()
            return

        query = message.content.strip().lower()
        if not query:
            await cl.Message(content="❗ Please enter a question or upload your resume.").send()
            return

        matched_support = match_support_service(query)
        if matched_support:
            keyword, url = matched_support
            await cl.Message(content=f"🔗 Here’s what you’re looking for:\n**{keyword.title()}**: {url}").send()
            return

        programs = cl.user_session.get("programs", [])
        resume_text = cl.user_session.get("resume_text", "")

        matches = []

        # 1. Query keyword matching
        for p in programs:
            score_title = fuzz.partial_ratio(query, p["title"].lower())
            score_description = fuzz.partial_ratio(query, p.get("course_description", "").lower())
            score_campus = max([fuzz.partial_ratio(query, c["campus"].lower()) for c in p.get("campus_codes", [])] + [0])

            total_score = score_title * 0.5 + score_description * 0.3 + score_campus * 0.2
            if total_score > 50:
                matches.append({"title": p["title"], "score": total_score, "details": p})

        # 2. Resume fallback using semantic similarity
        if not matches and resume_text:
            semantic_matches = semantic_program_match(resume_text, programs)
            for prog, sim_score in semantic_matches:
                matches.append({"title": prog["title"], "score": sim_score * 100, "details": prog})

        if matches:
            response = "🎓 Here are matching programs based on your query or resume:\n\n"
            for match in matches[:5]:
                p = match["details"]
                response += f"**{p['title']}** (Score: {match['score']:.1f})\n"
                response += f"Course Code: {p['course_code']}\n"
                response += f"Campus: {', '.join(c['campus'] for c in p['campus_codes'])}\n"
                response += f"Start Date: {p['start_date']}\n"
                response += f"Duration: {p['duration']}\n"
                response += f"Career Opportunities: {p['career_opportunities']}\n"
                response += f"Admission Requirements: {p['admission_requirements']}\n\n"
            await cl.Message(content=response).send()
        else:
            await cl.Message(content="❌ No programs matched your query or resume.").send()

    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {e}").send()
