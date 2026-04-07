# Free Model Setup Guide

## Problem
You've depleted your API credits and got error 402.

## Solutions

### 1️⃣ **QUICK FIX: Add Delay (5 min)**
Increase `REQUEST_DELAY` in `.env`:
```
REQUEST_DELAY=5.0
```
This adds 5 seconds between API requests, reducing costs significantly.

---

### 2️⃣ **BEST OPTION: Use Ollama (20 min, Completely FREE)**

**Install Ollama:**
1. Download from https://ollama.ai
2. Install and open the app

**Pull a model:**
```bash
ollama pull mistral
```

**Keep Ollama running:**
```bash
ollama serve
```
(Keep this terminal open)

**Update your `.env`:**
```
API_BASE_URL="http://localhost:11434/v1"
MODEL_NAME="mistral"
HF_TOKEN="ollama"
REQUEST_DELAY=1.0
```

**Test:**
```bash
cd /Users/harrymacbook/Desktop/hackenv
python3 inference.py
```

**Ollama Model Options:**
- `mistral` - Fast, good quality (recommended)
- `llama2` - Slower but popular
- `neural-chat` - Balanced
- `dolphin-mixtral` - Powerful but slow
- `orca-mini` - Lightweight

```bash
# Install different sizes based on your disk space:
ollama pull mistral         # ~4GB (recommended)
ollama pull neural-chat     # ~4GB
ollama pull llama2          # ~4GB
```

---

### 3️⃣ **Alternative: Use Groq (Free, 25 req/min)**

**Sign up:**
1. Go to https://console.groq.com
2. Create account
3. Get API key from dashboard

**Update `.env`:**
```
API_BASE_URL="https://api.groq.com/openai/v1"
HF_TOKEN="gsk_your_groq_api_key_here"
MODEL_NAME="mixtral-8x7b-32768"
REQUEST_DELAY=2.4  # 25 req/min = 60/25 = 2.4 sec per request
```

**Test:**
```bash
python3 inference.py
```

---

### 4️⃣ **Alternative: Use Together AI**

**Sign up:**
1. Go to https://api.together.xyz
2. Create account & get API key

**Update `.env`:**
```
API_BASE_URL="https://api.together.xyz/v1"
HF_TOKEN="your_together_api_key"
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
REQUEST_DELAY=2.0
```

---

## Cost Comparison

| Service | Cost | Setup Time | Limit |
|---------|------|-----------|-------|
| Ollama | FREE | 20 min | None (local) |
| Groq | FREE | 5 min | 25 req/min |
| Together AI | FREE tier | 5 min | Limited tokens |
| HuggingFace | Variable | - | Per credits |

---

## Recommendation
**Use Ollama** if you have disk space (~4-10GB)
- Completely free
- No rate limits
- Works offline
- Full control

If disk space is limited: **Use Groq** (fastest setup, free tier)

---

## Troubleshooting

**Ollama: "Connection refused"**
- Make sure `ollama serve` is running in another terminal

**Groq: "Rate limit exceeded"**
- Increase `REQUEST_DELAY=3.0` in your `.env`

**Model too slow?**
- Use `ollama/mistral` instead of larger models
- Or reduce `MAX_TOKENS` in inference.py (default: 300)

---

## Emergency: Keep API Running with High Delay

If you want to stick with your current API:

```env
REQUEST_DELAY=10.0    # 10 seconds between requests
# This reduces API calls to 6 per minute
```

Also reduce in `inference.py`:
```python
MAX_STEPS = 20  # Instead of 50
MAX_TOKENS = 100  # Instead of 300
```

This can reduce costs by 75% while still having a working system!
