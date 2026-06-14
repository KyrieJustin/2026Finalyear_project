
# A Multi-Tool Multimodal Agent for Fine-Grained Visual Understanding

This project aims to bridge the "Granularity Gap" in modern Vision-Language Models (VLMs). By integrating spatial grounding tools and real-world knowledge APIs, the system transforms a monolithic VLM into an evidence-driven reasoning agent capable of expert-level fine-grained visual classification.

## 🚀 Key Features

- **Differential Feature Pinpointing**: Leverages **Grounding-DINO** to programmatically crop high-resolution discriminative regions, overcoming the VLM's information bottleneck.
- **Knowledge-Augmented Reasoning**: Integrates **Wikipedia** and **Google Serper APIs** to retrieve real-time taxonomic facts, effectively mitigating object hallucinations.
- **Interpretable Reasoning Chain**: A multi-stage pipeline (Hypothesis -> Grounding -> Synthesis) that provides a transparent evidence chain for every conclusion.
- **Experimental Suite**: Includes full scripts for baseline testing, ablation studies, and deep error diagnosis.
- **Interactive Web GUI**: A minimalist interface built with **Gradio** for real-time demonstration.

---

## 🛠️ System Architecture

The agent operates through a three-stage closed-loop cycle:
1.  **Hypothesis Generation**: The VLM makes an initial coarse-grained guess to generate search keywords.
2.  **Data Acquisition & Pinpointing**: The agent fetches external knowledge while simultaneously zooming into key visual features using Grounding-DINO.
3.  **Synthesized Conclusion**: The VLM synthesizes the original context, high-res crops, and retrieved facts to produce a verified final answer.

---

## 📦 Installation & Setup

### 1. Environment Requirements
- Python 3.10+
- CUDA 12.1+ (Tested on NVIDIA H20)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# If not present, use:
pip install torch torchvision transformers accelerate qwen-vl-utils gradio requests wikipedia google-search-results
```

### 3. API Configuration
Open `real_tools.py` and insert your API keys:
```python
SERPER_API_KEY = "your_serper_api_key_here"
```

---

## 📊 Experimental Results

Evaluated on **FGExpertBench** (a custom dataset of 120 complex fine-grained tasks):

| Configuration | Overall Accuracy | Object ID Acc. | Gain |
| :--- | :---: | :---: | :---: |
| **Baseline (Vanilla Qwen2.5-VL)** | 78.33% | 69.84% | - |
| **Full Multi-Tool Agent** | **88.33%** | **88.89%** | **+10.0%** |

*Note: The agent demonstrates a significant **19.1% improvement** in the critical Object Identification category.*

---

## 📂 File Structure

### Core Logic
- `real_agent.py`: The primary agent implementation with multi-tool orchestration.
- `real_tools.py`: Utility class for Wikipedia and Serper API integration.
- `fgvc_model.py` / `fgvc_prompt.py`: Low-level model configurations and prompt templates.

### Execution Scripts
- `web_gui.py`: Launch the Gradio-based interactive demonstration.
- `run_real_agent.py`: Execute the full agent pipeline on the benchmark.
- `run_baseline.py`: Run the unaugmented baseline for comparison.
- `run_ablation.py`: Conduct systematic ablation studies (Vision-only vs. Knowledge-only).

### Evaluation & Analysis
- `evaluate.py`: Statistical scoring script for all result JSON files.
- `diagnose_errors.py`: Deep diagnosis tool that re-runs failed cases to capture reasoning logs.
- `error_case_study/`: Visual and textual logs of representative failure modes.
- `question.json`: The standard dataset definition.

---

## 🖥️ Usage

### To Launch the Interactive Demo:
```bash
python web_gui.py
```
Then open the provided local or public URL to upload images and ask questions.

### To Replicate Full Benchmarking:
```bash
python run_real_agent.py
python evaluate.py -f result_real_agent.json
```

---

## 🎓 Acknowledgments
Developed as a Final Year Project for the **BUPT-QMUL Joint Programme**. Special thanks to my supervisor for the guidance on fine-grained visual understanding and agentic workflows.

---

