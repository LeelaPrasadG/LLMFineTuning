# Vitalis AI: Brand-Voice Alignment via QLoRA Fine-Tuning

This repository provides an end-to-end, hardware-optimized pipeline for Parameter-Efficient Fine-Tuning (PEFT) using the QLoRA (Quantized Low-Rank Adaptation) methodology. The project takes a general-purpose instruction model (`Llama-3.1-8B-Instruct`) and modifies its conversational style to align with a specialized brand persona. By leveraging memory-efficient configurations and dynamic script fixes, the pipeline trains a custom corporate voice—specifically designed as an empathetic, jargon-free health insurance guide—directly on free-tier cloud hardware before compiling it for offline, local deployment.

---

### Phase 1: Environment Safety Initialization
* **Flush Cache:** Clear trailing data from memory using Python's garbage collector (`gc.collect()`) and PyTorch's empty cache command to maximize free workspace.
* **Set VRAM Cap:** Configure `gpu_memory_utilization = 0.3` to mandate a large, strict safety buffer on the cloud GPU, protecting the session from sudden out-of-memory crashes.

### Phase 2: Base Model Loading & Parameter-Efficient Rigging
* **Load 4-bit Base:** Stream the pre-shrunk `Llama-3.1-8B` base model from Hugging Face into memory, dropping its resource footprint from 16 GB down to ~5.5 GB.
* **Rig LoRA Adapters:** Freeze the foundational language layers and attach tiny, empty, trainable adapter matrices (`r = 16`) exclusively to the internal model attention modules.

### Phase 3: Dataset Ingestion & Tokenizer Mapping
* **Ingest Data:** Read the locally generated `vitalis_brand_alignment_1k.jsonl` file containing your 1,000 health insurance dialogue lines.
* **Map Chat Template:** Apply the native Meta Llama 3 structural chat tokens (`tokenizer.apply_chat_template`) across the dataset so the network explicitly understands where system instructions, customer queries, and target brand responses begin and end.

### Phase 4: Supervised Fine-Tuning Execution
* **Intercept Library Bug:** Run a custom Python loop that interceptively deletes modern deprecated parameters (`push_to_hub_token`, etc.) to prevent library version mismatches from breaking the runner.
* **Run Backpropagation:** Initialize the SFTTrainer in hardware-accelerated `fp16` precision to pass data over the GPU, tracking the descending `loss` value as the empty LoRA layers optimize their math to favor your warm, jargon-free brand voice.

### Phase 5: GGUF Model Export & Local Migration
* **Merge Weights:** Bake the newly mastered tone adapters directly back into the core model parameters using Unsloth's `save_pretrained_gguf`.
* **Compress & Download:** Save the combined network as a lightweight, 4-bit compressed `.gguf` file and download it to your hard drive so it is ready to execute 100% offline using your laptop's 32 GB of system RAM via Ollama.
