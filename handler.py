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
# Load SUMMARY model (T5)
# =====================================================
def load_summary_model():
    global summary_tokenizer, summary_model
    if summary_model is not None:
        return

    log("Loading SUMMARY model (T5)")
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
# Load TRANSLATION model (Marian)
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
# Detect pure separator lines (---, ___)
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

# =====================================================
# Translate text (TABLES INCLUDED, STRUCTURE SAFE)
# =====================================================
def translate_text(text: str) -> str:
    lines = text.split("\n")
    out_lines = []

    for line in lines:
        stripped = line.strip()

        # =================================================
        # 🔒 HARD GUARDS — DO NOT TRANSLATE MEANINGLESS INPUT
        # =================================================

        # Empty line
        if not stripped:
            out_lines.append(line)
            continue

        # Bullet / symbol-only line (•, -, *, etc.)
        if re.match(r"^[\u2022•\-\*\u00B7]+$", stripped):
            out_lines.append(line)
            continue

        # Very low linguistic content → skip
        if len(re.findall(r"[A-Za-zА-Яа-я]", stripped)) < 2:
            out_lines.append(line)
            continue

        # Full separator row (tables)
        if re.match(r"^\|\s*[-\s_\.]+\|\s*[-\s_\.]+\|\s*$", line):
            out_lines.append(line)
            continue

        # Pure separator (no pipes)
        if is_layout_line(line):
            out_lines.append(line)
            continue

        # =================================================
        # TABLE ROW
        # =================================================
        if "|" in line:
            cells = line.split("|")
            new_cells = []

            for cell in cells:
                cell_text = cell.strip()

                # Skip empty / separator cells
                if not cell_text or re.match(r"^[-\s_\.]+$", cell_text):
                    new_cells.append(cell)
                    continue

                # Skip non-linguistic cells
                if len(re.findall(r"[A-Za-zА-Яа-я]", cell_text)) < 2:
                    new_cells.append(cell)
                    continue

                inputs = translate_tokenizer(
                    cell_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(translate_model.device)

                with torch.no_grad():
                    output = translate_model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False
                    )

                translated = translate_tokenizer.decode(
                    output[0], skip_special_tokens=True
                )

                new_cells.append(f" {translated} ")

            out_lines.append("|".join(new_cells))
            continue

        # =================================================
        # NORMAL TEXT LINE
        # =================================================
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
# Summarize ALL pages together
# =====================================================
def clean_ocr_noise(text: str) -> str:
   
    cleaned_lines = []
    seen = set()

    for raw_line in text.split("\n"):
        line = raw_line.strip()

        # -------------------------------
        # Пустые строки
        # -------------------------------
        if not line:
            continue

        upper = line.upper()

        # -------------------------------
        # Подписи, свидетели, boilerplate
        # -------------------------------
        if upper.startswith("EXECUTED AS A DEED"):
            continue
        if upper.startswith("SIGNATURE OF WITNESS"):
            continue
        if upper.startswith("IN THE PRESIDENCY OF"):
            continue
        if re.match(r"^NAME:\s*$", upper):
            continue
        if re.match(r"^ADDRESS:\s*$", upper):
            continue

        # -------------------------------
        # OCR / редакторский мусор
        # -------------------------------
        if "SYNC, CORRECTED BY" in upper:
            continue
        if upper.startswith("==") and upper.endswith("=="):
            continue

        # -------------------------------
        # Плейсхолдеры
        # -------------------------------
        if line in {"[-]", "[ ]", "[__]", "_____"}:
            continue

        # -------------------------------
        # Линии разметки (----, ____ и т.п.)
        # -------------------------------
        if re.match(r"^[\-\._\s]{5,}$", line):
            continue

        # -------------------------------
        # Очень низкая языковая нагрузка
        # (оставляем заголовки, убираем мусор)
        # -------------------------------
        if len(re.findall(r"[A-Za-zА-Яа-я]", line)) < 5:
            continue

        # -------------------------------
        # Удаление дубликатов строк
        # -------------------------------
        key = upper
        if key in seen:
            continue
        seen.add(key)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def summarize_all_pages(pages):
    full_text = "\n\n".join(
        cleaned
        for p in pages
        if (cleaned := clean_ocr_noise(p["text"]))
        and len(re.findall(r"[A-Za-zА-Яа-я]", cleaned)) > 20
     )


    if not full_text.strip():
        return ""

    prompt = (
    "Rewrite the contract below in English. IMPORTANT: - This is NOT a summary. - You MUST restate ALL factual information in full. - Output MUST be at least 500 words. - Expand each section into detailed legal sentences.- Do NOT omit names, dates, addresses, amounts, percentages, or penalties.- If information appears in tables, convert it to full sentences.Write each required section as a separate heading."
    + full_text
     )


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

    return summary_tokenizer.decode(output[0], skip_special_tokens=True)

# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")

    pages = event["input"]["pages"]

    load_summary_model()
    load_translate_model()

    # ---- Summary ----
    log("Creating summary")
    raw_summary = summarize_all_pages(pages)
    summary = translate_text(raw_summary) if raw_summary else ""

    # ---- Translation ----
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
