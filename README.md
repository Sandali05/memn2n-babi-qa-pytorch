## 🧠 MemN2N for bAbI (PyTorch)
A clean, fully reproducible PyTorch implementation of the End-to-End Memory Network (MemN2N) for the Facebook bAbI Question Answering benchmark (v1.2).

🧰 Implements:
- Multi-hop attention over memory slots
- Position Encoding (Sukhbaatar et al., 2015)
- Optional trainable Temporal Encoding per memory slot
- Adjacent weight tying (A_{k+1} = C_k)
- Padding-safe embeddings
- Easy-to-use bAbI data pipeline (auto-download + parsing)

🧩 Features
 ✅ Multi-hop attention mechanism
 ✅ Position & Temporal encodings
 ✅ Clean reproducible training script
 ✅ Automatic dataset download (bAbI v1.2)
 ✅ Works with both en and en-10k versions
 ✅ Configurable memory size, hops, embedding dim, dropout, etc.