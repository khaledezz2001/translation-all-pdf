import time
import re
import torch
import runpod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MarianTokenizer,
    MarianMTModel,
    BitsAndBytesConfig,
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
# Load SUMMARY model (Qwen 2.5 14B – 4bit)
# =====================================================
def load_summary_model():
    global summary_tokenizer, summary_model
    if summary_model is not None:
        return

    log("Loading SUMMARY model (Qwen-2.5-14B-Instruct, 4-bit)")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    summary_tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    summary_model = AutoModelForCausalLM.from_pretrained(
        SUMMARY_MODEL_PATH,
        quantization_config=quant_config,
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

def chunk_text(text, max_tokens=3000):
    tokens = summary_tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield summary_tokenizer.decode(tokens[i:i + max_tokens])

# =====================================================
# Translation (structure safe)
# =====================================================
def translate_text(text: str) -> str:
    lines = text.split("\n")
    out = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            out.append(line)
            continue
        if is_layout_line(line):
            out.append(line)
            continue
        if len(re.findall(r"[A-Za-zА-Яа-я]", stripped)) < 2:
            out.append(line)
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

        out.append(
            translate_tokenizer.decode(output[0], skip_special_tokens=True)
        )

    return "\n".join(out)

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
        if re.match(r"^[\-\._\s]{5,}$", line):
            continue
        if len(re.findall(r"[A-Za-zА-Яa-я]", line)) < 5:
            continue
        if upper in seen:
            continue

        seen.add(upper)
        cleaned.append(line)

    return "\n".join(cleaned)

# =====================================================
# Summarize / Rewrite (ENGLISH ONLY, chunked)
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
        "Rewrite the contract in English.\n"
        "Rules:\n"
        "- This is NOT a summary\n"
        "- Restate ALL factual information\n"
        "- Do NOT omit names, dates, addresses, amounts, penalties\n"
        "- Expand into formal legal language\n"
        "- Convert tables into sentences\n\n"
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
                max_new_tokens=900,
                do_sample=False
            )

        text = summary_tokenizer.decode(
            output[0], skip_special_tokens=True
        )

        outputs.append(text)

    return "\n\n".join(outputs)

# =====================================================
# RunPod handler (CORRECT ORDER)
# =====================================================
def handler(event):
    log("Handler started")

    pages = event["input"]["pages"]

    load_translate_model()
    load_summary_model()

    # 1️⃣ Translate FIRST
    log("Translating pages to English")
    for p in pages:
        p["text"] = translate_text(p["text"])

    # 2️⃣ Summarize / rewrite English text
    log("Creating summary from English text")
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
