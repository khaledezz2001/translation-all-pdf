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
# Helpers
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

def chunk_text(text, max_tokens=2800):
    tokens = summary_tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield summary_tokenizer.decode(tokens[i:i + max_tokens])

# =====================================================
# STRUCTURE-SAFE TRANSLATION
# =====================================================
def translate_text(text: str) -> str:
    lines = text.split("\n")
    out_lines = []

    for line in lines:
        stripped = line.strip()

        # Empty line
        if not stripped:
            out_lines.append(line)
            continue

        # Table separator row
        if re.match(r"^\|\s*[-\s_]+\|", line):
            out_lines.append(line)
            continue

        # Table row → translate cells only
        if "|" in line:
            cells = line.split("|")
            new_cells = []

            for cell in cells:
                cell_text = cell.strip()

                if not cell_text or len(re.findall(r"[A-Za-zА-Яа-я]", cell_text)) < 2:
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

        # Non-linguistic line
        if len(re.findall(r"[A-Za-zА-Яа-я]", stripped)) < 2:
            out_lines.append(line)
            continue

        # Normal text line
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
# SUMMARY (~100 WORDS)
# =====================================================
def summarize_all_pages(pages):
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
        "Summarize the contract in clear English.\n"
        "Rules:\n"
        "- This MUST be a concise summary, not a rewrite\n"
        "- Length MUST be about 100 words (not more than 120)\n"
        "- Include the parties, date, location, and purpose\n"
        "- Do NOT invent clauses or sections\n"
        "- Ignore table formatting and layout symbols\n"
        "- Do NOT include signatures or boilerplate\n\n"
    )

    outputs = []

    for chunk in chunk_text(full_text):
        prompt = (
            "<|system|>\n" + system_prompt +
            "<|user|>\n" + chunk +
            "\n<|assistant|>\n"
        )

        inputs = summary_tokenizer(
            prompt,
            return_tensors="pt"
        ).to(summary_model.device)

        with torch.no_grad():
            output = summary_model.generate(
                **inputs,
                max_new_tokens=180,
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

    load_translate_model()
    load_summary_model()

    # 1️⃣ Translate first (structure preserved)
    log("Translating pages to English")
    for p in pages:
        p["text"] = translate_text(p["text"])

    # 2️⃣ Summarize (~100 words)
    log("Creating summary")
    summary = summarize_all_pages(pages)

    log("Handler finished")

    return {
        "summary": summary,
        "pages": pages
    }

# =====================================================
# Start RunPod serverless
# =====================================================
runpod.serverless.start({"handler": handler})
