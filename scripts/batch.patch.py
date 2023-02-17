from modules import scripts, processing

import gradio as gr


def swap_img2img_pipeline(p: processing.StableDiffusionProcessingImg2Img):
    p.__class__ = processing.StableDiffusionProcessingTxt2Img
    dummy = processing.StableDiffusionProcessingTxt2Img()
    for k, v in dummy.__dict__.items():
        if hasattr(p, k):
            continue
        setattr(p, k, v)


class ControlNetBatchPatch(scripts.Script):
    def title(self):
        return "ControlNet batch patch"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group(visible=is_img2img):
            with gr.Accordion("ControlNet Batch Patch", open=False):
                enabled = gr.Checkbox(label="Enabled")
        return [enabled]

    def process(self, p, enabled):
        if enabled:
            swap_img2img_pipeline(p)
