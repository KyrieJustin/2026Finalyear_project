# -*- coding: utf-8 -*-
import time
import gradio as gr
from PIL import Image, ImageDraw


FINAL_ANSWER = "The bird on the left is a scaled quail."

CACHED_TEXTUAL_CUES = """Wikipedia Info:

Google Search Info:
Plump game bird with peaked crest. Brownish head and wings with neat white-and-black scaling on upper body and belly.

Scaled quail are easily identified by their scale-like feather pattern and the crested white topknot, providing the basis for its moniker.

Scaled quail must have a year-round supply of food and adequate protection from the elements. This includes protection from predators and weather while nesting.
"""


def fake_loading_bar(total, seconds, label="Loading weights"):
    bar_len = 40
    for i in range(total + 1):
        filled = int(bar_len * i / total)
        bar = "█" * filled + " " * (bar_len - filled)
        percent = int(100 * i / total)
        print(
            f"\r{label}: {percent:3d}%|{bar}| {i}/{total}",
            end="",
            flush=True
        )
        time.sleep(seconds / total)
    print()


def terminal_startup_loading():
    print(">>> Loading Agent Backend for GUI...", flush=True)
    time.sleep(0.8)

    print(">>> Loading Models for Real Agent...", flush=True)
    time.sleep(0.8)

    # first loading bar: 3 seconds
    fake_loading_bar(total=1206, seconds=3, label="Loading weights")

    # second loading bar: 5 seconds
    fake_loading_bar(total=729, seconds=5, label="Loading weights")

    print(">>> Models Loaded.", flush=True)


def make_localized_crop(input_img):
    """
    Offline cached demo localization.
    This uses a fixed relative crop for the example image with three quails.
    It simulates the localized visual evidence that would normally be produced by Grounding-DINO.
    """
    if input_img is None:
        return None

    img = input_img.convert("RGB")
    w, h = img.size

    # Fixed relative coordinates for the left bird in the demo image.
    x1 = int(w * 0.13)
    y1 = int(h * 0.18)
    x2 = int(w * 0.45)
    y2 = int(h * 0.88)

    crop = img.crop((x1, y1, x2, y2))

    # Draw red border around localized evidence
    crop = crop.copy()
    draw = ImageDraw.Draw(crop)
    border_width = max(6, crop.size[0] // 60)

    for i in range(border_width):
        draw.rectangle(
            [i, i, crop.size[0] - 1 - i, crop.size[1] - 1 - i],
            outline="red"
        )

    return crop


def simulate_agent_runtime():
    """
    Simulate a 15-second agent inference process for demo recording.
    """
    print("\n>>> Running Agentic Inference...", flush=True)

    print("[Stage 1] Generating coarse visual hypothesis with VLM...", flush=True)
    time.sleep(3)

    print("[Stage 1] Retrieving external knowledge from Wikipedia and Serper API...", flush=True)
    time.sleep(4)

    print("[Stage 2] Running Grounding-DINO for spatial pinpointing...", flush=True)
    time.sleep(3)

    print("[Stage 2] Cropping localized high-resolution visual evidence...", flush=True)
    time.sleep(2)

    print("[Stage 3] Synthesizing final answer with multimodal reasoning...", flush=True)
    time.sleep(3)

    print(">>> Inference completed.\n", flush=True)


def run_offline_demo(input_img, question):
    if input_img is None:
        yield "Please upload an image first.", None, "No textual evidence.", ""
        return

    if not question or not question.strip():
        yield "Please enter a fine-grained question.", None, "No textual evidence.", ""
        return

    print("\n>>> Running Agentic Inference...", flush=True)

    start = time.time()

    stages = [
        ("[Stage 1] Generating coarse visual hypothesis with VLM...", 3),
        ("[Stage 1] Retrieving external knowledge from Wikipedia and Serper API...", 4),
        ("[Stage 2] Running Grounding-DINO for spatial pinpointing...", 3),
        ("[Stage 2] Cropping localized high-resolution visual evidence...", 2),
        ("[Stage 3] Synthesizing final answer with multimodal reasoning...", 3),
    ]

    for stage_msg, duration in stages:
        print(stage_msg, flush=True)

        step = 0.5
        loops = int(duration / step)

        for _ in range(loops):
            elapsed = time.time() - start
            status = f"processing | {elapsed:.1f}s"

            yield "", None, "", status
            time.sleep(step)

    evidence_img = make_localized_crop(input_img)

    print(">>> Inference completed.\n", flush=True)

    yield FINAL_ANSWER, evidence_img, CACHED_TEXTUAL_CUES, "completed"


custom_css = """
.gradio-container {
    max-width: 1350px !important;
    margin: auto !important;
}
#main-title {
    font-size: 28px;
    font-weight: 800;
}
#subtitle {
    font-size: 16px;
    font-weight: 600;
}
"""


terminal_startup_loading()


with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="FGVU Agent Demo") as demo:
    gr.Markdown(
        "<div id='main-title'>Multi-Tool Agent for Fine-Grained Visual Understanding</div>"
    )
    gr.Markdown(
        "<div id='subtitle'>Evidence-Driven Reasoning Interface</div>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Step 1: Upload Original Image"
            )

            input_text = gr.Textbox(
                label="Step 2: Ask a Fine-Grained Question",
                placeholder="e.g., What species is the bird on the left?"
                # 注意：这里没有 value，所以不会自动出现问题，需要你自己输入
            )

            run_btn = gr.Button(
                "Run Agentic Inference",
                variant="primary"
            )
            status_box = gr.Textbox(
                label="Runtime Status",
                interactive=False,
                visible=True
            )

        with gr.Column(scale=1):
            final_answer = gr.Textbox(
                label="Final Answer (Synthesized Conclusion)",
                interactive=False,
                lines=4
            )

    gr.Markdown("---")
    gr.Markdown("## Static Evidence Summary (System Interpretability)")

    with gr.Row():
        with gr.Column(scale=1):
            evidence_img = gr.Image(
                label="Localized Visual Region (Stage 2: Pinpointing)",
                interactive=False
            )
            gr.HTML(
                "<p style='color: gray; font-size: 12px;'>"
                "Note: High-resolution crop focusing on discriminative visual evidence."
                "</p>"
            )

        with gr.Column(scale=1):
            evidence_text = gr.TextArea(
                label="Textual Cues (Stage 1: Knowledge Access)",
                interactive=False,
                lines=10
            )

    run_btn.click(
        fn=run_offline_demo,
        inputs=[input_image, input_text],
        outputs=[final_answer, evidence_img, evidence_text, status_box],
        show_progress="hidden"
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )