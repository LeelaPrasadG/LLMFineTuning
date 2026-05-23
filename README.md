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

# Vitalis AI: Brand-Voice Alignment via QLoRA Fine-Tuning

This repository provides an end-to-end, hardware-optimized pipeline for Parameter-Efficient Fine-Tuning (PEFT) using the QLoRA (Quantized Low-Rank Adaptation) methodology. The project takes a general-purpose instruction model (`Llama-3.1-8B-Instruct`) and modifies its conversational style to align with a specialized brand persona. By leveraging memory-efficient configurations and dynamic script fixes, the pipeline trains a custom corporate voice—specifically designed as an empathetic, jargon-free health insurance guide—directly on free-tier cloud hardware before compiling it for offline, local deployment.

---

### Step 1: Switch Colab to the T4 GPU Runtime
* Navigate to the top toolbar menu in Google Colab and click **Runtime**.
* Select **Change runtime type**.
* Under *Hardware accelerator*, select **T4 GPU** and click **Save** *(Note: If your workspace refreshes and wipes files, simply rerun your previous dataset generator cell to place your `.jsonl` file back into `/content/`)*.

### Step 2: Install Unsloth and Dependencies
* Run the targeted environment-hardening installation commands inside your notebook to pull down the optimized compiling layers (`unsloth_zoo`, `unsloth[colab-new]`, `xformers`, and `trl`).

### Step 3: Load the Model and Apply LoRA Configuration
* Execute the hardware-railed loading pipeline using `FastLanguageModel.from_pretrained` to load the 4-bit base model (`meta-Llama-3.1-8B-Instruct-bnb-4bit`).
* Lock down the safety rails by setting `gpu_memory_utilization = 0.3` and tightening `max_seq_length = 512` before targeting the attention modules with the LoRA adapter rigging layer (`r = 16`).

### Step 4: Parse and Format the Dataset
* Ingest your local health insurance training file (`vitalis_brand_alignment_1k.jsonl`).
* Apply the structural chat template mapping function (`tokenizer.apply_chat_template`) across all 1,000 dialogue pairs to match the native Meta Llama 3 bracketed token configuration.

### Step 5: Execute Supervised Fine-Tuning (SFT)
* Initialize the script's custom attribute deletion loop to clear modern deprecated parameters (`push_to_hub_token`, etc.) and completely bypass version conflicts between `trl` and `transformers`.
* Run `trainer.train()` using accelerated `fp16=True` precision and track the descending `loss` metric table as the empty adapters study your custom brand tone.

### Step 6: Test and Verify the Aligned Tone
* Put the fine-tuned network into inference mode using `FastLanguageModel.for_inference(model)`.
* Run a test customer prompt through the model to verify that it responds in the plain-spoken, empathetic style of Vitalis AI instead of spitting out rigid, bureaucratic corporate jargon.

### Step 7: Export to GGUF and Download to Your Laptop
* Merge the checkpoint-60 adapter adjustments directly back into the core model weights and quantize them into a local-friendly 4-bit file package by running `model.save_pretrained_gguf("vitalis_brand_model", tokenizer, quantization_method = "q4_k_m")`.
* Download the resulting ~4.8 GB `Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf` asset directly to your local computer's storage space *(Note: Use Google Drive mounting or network tunnels if your browser downloader encounters size timeouts)*.

### Step 8: Open Your Laptop Terminal and Check Ollama
* Ensure the local runtime engine is running on your laptop.
* Open your system command prompt or terminal window and execute `ollama --version` to confirm that the environment is fully initialized.

### Step 9: Create and Configure Your Local Modelfile
* Move the downloaded `.gguf` file out of your system Downloads folder and paste it into a dedicated project directory (e.g., `vitalis-ai`).
* Inside that same directory, create a plain text file named exactly `Modelfile` and construct your system deployment blueprint mapping the base `.gguf` path, the core system instruction prompt, and target parameter constraints (`temperature 0.3`, `top_p 0.9`).

### Step 10: Register and Run Vitalis AI Locally
* Navigate directly into your project directory using the terminal `cd` command (e.g., `cd path/to/vitalis-ai`).
* Register your structural custom blueprint package by executing:
  ```bash
  ollama create vitalis-ai -f Modelfile

### Launch a secure, ultra-low latency, 100% offline chat instance running directly inside your laptop's 32 GB of system RAM by executing:
```bash
  ollama run vitalis-ai

### 🎯 Conclusion

The exported `Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf` file represents the complete, standalone brain of your custom language model. Every single word of the response is calculated and generated directly from this self-contained file. To understand exactly how this works on your local machine, think of it as a video game console vs. a video game disc:

1. **The GGUF File is the "Game Disc" (The Knowledge & Identity):** This file is the static, self-contained mathematical brain of your AI model. It holds all 8 billion parameters that Meta originally trained. Crucially, because your script ran `model.load_adapter("outputs/checkpoint-60")` right before exporting, your custom fine-tuned brand adjustments are permanently baked directly into this file. It contains the actual math that shifts the model's vocabulary away from strict corporate jargon and forces it to select warm, plain-English words.
2. **Ollama is the "Game Console" (The Engine):** Ollama does not contain any insurance knowledge or tone properties. It is simply the local hardware interpreter that opens your `.gguf` file, maps its internal structures into your machine's high-speed RAM, and passes the heavy mathematical execution threads directly onto your laptop's processors.

Every ounce of intelligence, empathy, and insurance knowledge comes directly out of that 4.8 GB file sitting on your hard drive. Ollama is just the engine turning those static numbers into a living conversation!