import time
import re
import torch
import runpod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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
SUMMARY_MODEL_PATH = "/models/hf/qwen"
TRANSLATE_MODEL_PATH = "/models/hf/marian-ru-en"

summary_tokenizer = None
summary_model = None
translate_tokenizer = None
translate_model = None

# =====================================================
# Load SUMMARY model (Qwen 2.5 7B – FP16)
# =====================================================
def load_summary_model():
    global summary_tokenizer, summary_model
    if summary_model is not None:
        return

    log("Loading SUMMARY model (Qwen-2.5-7B-Instruct, FP16)")

    summary_tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    summary_model = AutoModelForCausalLM.from_pretrained(
        SUMMARY_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
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

    log("Loading TRANSLATION model (Marian RU → EN)")

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
    log("TRANSLATION model loaded")

# =====================================================
# Detect layout separators
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

# =====================================================
# TRANSLATION (OLD, PROVEN, STRUCTURE-SAFE VERSION)
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

        if re.match(r"^\|\s*[-\s_\.]+\|\s*[-\s_\.]+\|\s*$", line):
            out_lines.append(line)
            continue

        if is_layout_line(line):
            out_lines.append(line)
            continue

        # ---------- TABLE ROW ----------
        if "|" in line:
            cells = line.split("|")
            new_cells = []

            for cell in cells:
                cell_text = cell.strip()

                if not cell_text or re.match(r"^[-\s_\.]+$", cell_text):
                    new_cells.append(cell)
                    continue

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

        # ---------- NORMAL LINE ----------
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
# OCR cleanup (used only for summary)
# =====================================================
def clean_ocr_noise(text: str) -> str:
    cleaned = []
    seen = set()

    for raw in text.split("\n"):
        line = raw.strip()
        upper = line.upper()

        if not line:
            continue
        if is_layout_line(line):
            continue
        if len(re.findall(r"[A-Za-z]", line)) < 5:
            continue
        if upper in seen:
            continue

        seen.add(upper)
        cleaned.append(line)

    return "\n".join(cleaned)

# =====================================================
# Word limiter
# =====================================================
def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

# =====================================================
# SUMMARY (N WORDS, CLEAN OUTPUT ONLY)
# =====================================================
def summarize_all_pages(pages, max_words: int):
    full_text = "\n\n".join(
        cleaned
        for p in pages
        if (cleaned := clean_ocr_noise(p["text"]))
        and len(re.findall(r"[A-Za-z]", cleaned)) > 20
    )

    if not full_text.strip():
        return ""

    system_prompt = (
        "You are a professional legal assistant.\n"
        "Summarize the document in clear English.\n"
        "Rules:\n"
        "- This MUST be a concise summary, not a rewrite\n"
        "- Do NOT invent facts or clauses\n"
        "- Include only key information\n"
        "- Ignore layout, tables, and formatting\n\n"
    )

    prompt = (
        "<|system|>\n" + system_prompt +
        "<|user|>\n" + full_text +
        "\n<|assistant|>\n"
    )

    inputs = summary_tokenizer(
        prompt,
        return_tensors="pt"
    ).to(summary_model.device)

    with torch.no_grad():
        output = summary_model.generate(
            **inputs,
            max_new_tokens=max_words * 2,
            min_new_tokens=max(30, max_words // 2),
            do_sample=False
        )

    decoded = summary_tokenizer.decode(
        output[0], skip_special_tokens=True
    )

    # Extract assistant response only
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1]

    decoded = re.sub(r"<\|.*?\|>", "", decoded).strip()

    return limit_words(decoded, max_words)

# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")

    input_data = event["input"]
    pages = input_data["pages"]

    # Read desired word count (default = 100)
    max_words = int(input_data.get("n_words", 100))

    load_translate_model()
    load_summary_model()

    # 1️⃣ Translate pages
    log("Translating pages")
    for p in pages:
        p["text"] = translate_text(p["text"])

    # 2️⃣ Summarize
    log(f"Creating summary ({max_words} words)")
    summary = summarize_all_pages(pages, max_words)

    log("Handler finished")

    return {
        "summary": summary,
        "pages": pages
    }

# =====================================================
# Start RunPod serverless
# =====================================================
runpod.serverless.start({"handler": handler})
