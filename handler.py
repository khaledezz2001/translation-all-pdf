import time
import re
import torch
import runpod

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianTokenizer,
    MarianMTModel,
)

# =====================================================
# Logging helper
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================
# Model paths
# =====================================================
SUMMARY_MODEL_PATH = "/models/hf/t5-summary"
TRANSLATE_MODEL_PATH = "/models/hf/marian-ru-en"

summary_tokenizer = None
summary_model = None
translate_tokenizer = None
translate_model = None

# =====================================================
# Load SUMMARY model (FLAN-T5-XL)
# =====================================================
def load_summary_model():
    global summary_tokenizer, summary_model
    if summary_model is not None:
        return

    log("Loading SUMMARY model (FLAN-T5-XL)")
    summary_tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL_PATH,
        local_files_only=True,
        use_fast=False
    )

    summary_model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARY_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    summary_model.eval()
    log("SUMMARY model loaded")

# =====================================================
# Load TRANSLATION model (Marian RU → EN)
# =====================================================
def load_translate_model():
    global translate_tokenizer, translate_model
    if translate_model is not None:
        return

    log("Loading TRANSLATION model (Marian)")
    translate_tokenizer = MarianTokenizer.from_pretrained(
        TRANSLATE_MODEL_PATH,
        local_files_only=True
    )

    translate_model = MarianMTModel.from_pretrained(
        TRANSLATE_MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to("cuda")

    translate_model.eval()
    log("TRANSLATION model loaded on cuda")

# =====================================================
# Helpers
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

def chunk_text(text, max_tokens=1800):
    tokens = summary_tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield summary_tokenizer.decode(tokens[i:i + max_tokens])

# =====================================================
# Translation (structure-safe)
# =====================================================
def translate_text(text: str) -> str:
    lines = text.split("\n")
    out_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            out_lines.append(line)
            continue

        if re.match(r"^[\u2022•\-\*\u00B7]+$", stripped):
            out_lines.append(line)
            continue

        if len(re.findall(r"[A-Za-zА-Яа-я]", stripped)) < 2:
            out_lines.append(line)
            continue

        if is_layout_line(line):
            out_lines.append(line)
            continue

        inputs = translate_tokenizer(
            line,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(translate_model.device)

        with torch.no_grad():
            output = translate_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        out_lines.append(
            translate_tokenizer.decode(output[0], skip_special_tokens=True)
        )

    return "\n".join(out_lines)

# =====================================================
# OCR cleanup
# =====================================================
def clean_ocr_noise(text: str) -> str:
    cleaned_lines = []
    seen = set()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        upper = line.upper()

        if not line:
            continue
        if upper.startswith(("EXECUTED AS A DEED", "SIGNATURE OF WITNESS")):
            continue
        if re.match(r"^[\-\._\s]{5,}$", line):
            continue
        if len(re.findall(r"[A-Za-zА-Яа-я]", line)) < 5:
            continue
        if upper in seen:
            continue

        seen.add(upper)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# =====================================================
# Summarize ALL pages (chunked)
# =====================================================
def summarize_all_pages(pages):
    full_text = "\n\n".join(
        cleaned
        for p in pages
        if (cleaned := clean_ocr_noise(p["text"]))
        and len(re.findall(r"[A-Za-zА-Яа-я]", cleaned)) > 20
    )

    if not full_text.strip():
        return ""

    prompt_prefix = (
        "Rewrite the contract below in English.\n"
        "IMPORTANT:\n"
        "- This is NOT a summary.\n"
        "- Restate ALL factual information in full.\n"
        "- Do NOT omit names, dates, addresses, amounts, or penalties.\n"
        "- Expand into formal legal sentences.\n"
        "- Convert tables into sentences.\n\n"
    )

    outputs = []

    for chunk in chunk_text(full_text):
        prompt = prompt_prefix + chunk

        inputs = summary_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(summary_model.device)

        with torch.no_grad():
            output = summary_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        outputs.append(
            summary_tokenizer.decode(output[0], skip_special_tokens=True)
        )

    return "\n\n".join(outputs)

# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")

    pages = event["input"]["pages"]

    load_summary_model()
    load_translate_model()

    log("Creating summary")
    raw_summary = summarize_all_pages(pages)
    summary = translate_text(raw_summary) if raw_summary else ""

    log("Translating pages")
    for p in pages:
        p["text"] = translate_text(p["text"])

    log("Handler finished")

    return {
        "summary": summary,
        "pages": pages
    }

# =====================================================
# Start RunPod serverless
# =====================================================
runpod.serverless.start({"handler": handler})
