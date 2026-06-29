# Clair-3B

Clair-3B is a highly capable 3-billion parameter language model designed for advanced conversational AI, coding assistance, and complex reasoning tasks.

## Model Details

- **Model Name:** Clair-3B
- **Parameters:** 3 billion
- **Architecture:** Transformer-based language model
- **Context Window:** 4,096 tokens
- **Format:** GGUF (F16)
- **Size:** 5.75 GB

## Key Features

Clair-3B delivers exceptional performance across a wide range of tasks:

- It possesses **significantly enhanced knowledge** and has greatly improved capabilities in **coding** and **mathematics**, due to specialized training in these domains.

- It demonstrates significant advancements in **instruction following**, **long-text generation**, **understanding structured data** (e.g., tables, JSON), and **generating structured outputs**, especially in JSON format. It is also **highly resilient to diverse system prompts**, improving role-play and condition-setting for chatbots.

- It supports **long contexts** of up to 4,096 tokens and can generate coherent, high-quality responses.

- It offers **multilingual support** for over 29 languages, including English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

### Core Capabilities

- 💬 **Natural Conversation**: Engaging and contextually aware dialogue
- 💻 **Code Assistance**: Code generation, explanation, debugging, and optimization
- 🧮 **Mathematical Reasoning**: Complex problem solving and step-by-step explanations
- 📝 **Text Generation**: Creative writing, summarization, and content creation
- 🌍 **Multilingual Support**: Fluent in 29+ languages
- 🎯 **Instruction Following**: Precise adherence to complex instructions and constraints
- 📊 **Structured Data**: Understanding and generating JSON, tables, and structured formats

## Installation

### Prerequisites
- [Ollama](https://ollama.com/download) installed on your system
- At least 6 GB of available RAM (8 GB recommended)
- Internet connection for initial download

### Quick Install

```bash
ollama pull r245142r/Clair-3B
```

### Manual Installation

If you prefer to use a local GGUF file:

1. Download the model file (5.75 GB)
2. Create a `Modelfile`:

```dockerfile
FROM ./clair-v4-float16.gguf

SYSTEM """You are Clair, a helpful and friendly AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 512
PARAMETER repeat_penalty 1.1
PARAMETER stop "\n\n"
PARAMETER stop "User:"
PARAMETER stop "Human:"
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 4096
PARAMETER num_gpu -1
```

3. Create the model:

```bash
ollama create clair -f Modelfile
```

## Usage

### Interactive Chat

```bash
ollama run r245142r/Clair-3B
```

Then start chatting:

```
>>> Can you help me with Python?
Of course! I'd be happy to help you with Python. What would you like to work on?

>>> Explain recursion with an example
Recursion is when a function calls itself to solve a problem. Here's a simple factorial example...

>>> Write a function to calculate fibonacci numbers
Here's an efficient fibonacci function using dynamic programming...
```

### API Usage

#### REST API

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "r245142r/Clair-3B",
  "prompt": "What is your name and who made you?"
}'
```

#### Chat API

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "r245142r/Clair-3B",
  "messages": [
    {
      "role": "user",
      "content": "Tell me about yourself"
    }
  ]
}'
```

### Python Integration

```python
import ollama

response = ollama.chat(
    model='r245142r/Clair-3B',
    messages=[
        {
            'role': 'user',
            'content': 'What is your name and who made you?'
        }
    ]
)

print(response['message']['content'])
```

### JavaScript/Node.js Integration

```javascript
import ollama from 'ollama';

const response = await ollama.chat({
  model: 'r245142r/Clair-3B',
  messages: [
    {
      role: 'user',
      content: 'What is your name and who made you?'
    }
  ]
});

console.log(response.message.content);
```

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `temperature` | 0.7 | Controls randomness (0.0-1.0) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 40 | Limits token selection |
| `num_predict` | 512 | Maximum tokens to generate |
| `repeat_penalty` | 1.1 | Penalizes repetitive text |
| `num_ctx` | 4096 | Context window size |
| `num_gpu` | -1 | GPU layers (-1 = all) |

### Customizing Parameters

You can override default parameters when running:

```bash
ollama run r245142r/Clair-3B --temperature 0.5 --num-predict 1024
```

Or in your Modelfile:

```dockerfile
PARAMETER temperature 0.5
PARAMETER num_predict 1024
```

## Context Window

Clair supports a **4,096 token context window**, which is approximately:
- 3,000 words of English text
- 10-15 pages of a typical document
- 50-100 lines of code

For longer conversations, consider:
- Summarizing previous context
- Starting a new conversation
- Using the `num_ctx` parameter to increase context (requires more RAM)

## Performance

### Hardware Requirements

| Configuration | RAM | GPU | Performance |
|---------------|-----|-----|-------------|
| Minimum | 6 GB | None | CPU-only, slower |
| Recommended | 8 GB | 4+ GB VRAM | GPU-accelerated |
| Optimal | 16 GB | 8+ GB VRAM | Fast inference |

### Speed Benchmarks

On typical hardware:
- **CPU-only:** 5-15 tokens/second
- **GPU-accelerated:** 30-60 tokens/second

## Prompting Best Practices

### For Best Results

1. **Be specific and clear** in your requests
2. **Provide context** when asking complex questions
3. **Use examples** to clarify your intent
4. **Break down complex tasks** into smaller steps

### Example Prompts

**Good:**
```
Can you explain how recursion works in Python with a simple example?
```

**Better:**
```
I'm learning Python and struggling with recursion. Can you explain it with a factorial function example and walk me through how it works step by step?
```

### System Prompts (Optional)

Clair-3B works excellently without system prompts, but you can use them to customize behavior for specific use cases:

```bash
ollama run r245142r/Clair-3B --system "You are a helpful coding tutor specializing in Python."
```

Or for different roles:

```bash
ollama run r245142r/Clair-3B --system "You are a mathematics professor explaining concepts to students."
```

## Troubleshooting

### Model Not Found

```bash
# Re-pull the model
ollama pull r245142r/Clair-3B
```

### Out of Memory

If you get OOM errors:

1. Close other applications
2. Reduce context window:
   ```bash
   ollama run r245142r/Clair-3B --num-ctx 2048
   ```
3. Use CPU-only mode:
   ```bash
   ollama run r245142r/Clair-3B --num-gpu 0
   ```

### Slow Performance

1. Ensure GPU acceleration is enabled
2. Close other GPU-intensive applications
3. Consider using a quantized version (Q4_K_M or Q5_K_M) for faster inference

### Model Not Responding Correctly

1. Try a fresh conversation
2. Clear Ollama cache:
   ```bash
   ollama rm r245142r/Clair-3B
   ollama pull r245142r/Clair-3B
   ```

## Technical Details

### Model Architecture

Clair-3B is built on a transformer architecture with:
- 3 billion parameters
- Optimized for conversational AI
- Fine-tuned for personality embedding

### Training Data

The model was trained on a diverse dataset including:
- Conversational data
- Technical documentation
- Code examples
- General knowledge
- Personality-specific examples

### Quantization Options

While this release uses F16 (full precision), quantized versions are available:

| Format | Size | Quality | Speed |
|--------|------|---------|-------|
| F16 | 5.75 GB | Best | Baseline |
| Q5_K_M | ~2.1 GB | Excellent | Faster |
| Q4_K_M | ~1.8 GB | Very Good | Fastest |
| Q3_K_M | ~1.5 GB | Good | Fastest |

## License and Usage

This model is provided for research and personal use. Please respect the creator's work and use responsibly.

## Credits

**Created by:** Michael Mlungisi Nkomo  
**Location:** Zimbabwe  
**Project:** Clair AI Assistant

## Support and Community

For issues, questions, or contributions:
- GitHub: [zim-my repository](https://github.com/Kedarcv/zim-my)
- Issues: Report bugs or request features on GitHub

## Changelog

### Version 4 (Current)
- ✅ Personality embedded in model weights
- ✅ Works without system prompts
- ✅ Improved identity consistency
- ✅ Better creator attribution
- ✅ F16 GGUF format for Ollama

### Version 3
- Initial LoRA-based implementation
- Required system prompts for personality
- Multiple quantization options

## Citation

If you use Clair-3B in your research or projects, please cite:

```bibtex
@misc{clair3b2026,
  author = {Michael Mlungisi Nkomo},
  title = {Clair-3B: An AI Assistant with Embedded Personality},
  year = {2026},
  publisher = {Ollama},
  url = {https://ollama.com/r245142r/Clair-3B}
}
```

---

**Note:** This model represents a novel approach to AI personality embedding through weight-level training rather than prompt engineering. The personality and identity are intrinsic to the model, not added through external prompts.
