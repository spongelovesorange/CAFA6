import sys

mods = [
    ("torch", "__version__"),
    ("transformers", "__version__"),
    ("huggingface_hub", "__version__"),
    ("tokenizers", "__version__"),
    ("accelerate", "__version__"),
    ("safetensors", "__version__"),
    ("sentencepiece", "__version__"),
]

print(f"python {sys.version}")
for name, attr in mods:
    try:
        m = __import__(name)
        print(f"{name} {getattr(m, attr, 'unknown')}")
    except Exception as e:
        print(f"{name} import failed: {e}")
