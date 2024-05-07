import gradio as gr
from modules import scripts, deepbooru
from modules.processing import process_images
import modules.shared as shared


"""

Thanks to Mathias Russ.
Thanks to RookHyena.

"""


class Script(scripts.Script):

    def title(self):
        return "Img2img batch interrogator"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        in_front = gr.Checkbox(label="Prompt in front", value=True)
        prompt_weight = gr.Slider(
            0.0, 1.0, value=0.5, step=0.1, label="interrogator weight"
        )
        use_weight = gr.Checkbox(label="Use Weighted Prompt", value=True)
        use_deepbooru = gr.Checkbox(label="Use deepbooru", value=True)
        return [in_front, prompt_weight, use_deepbooru, use_weight]

    def run(self, p, in_front, prompt_weight, use_deepbooru, use_weight):

        raw_prompt = p.prompt

        # fix alpha channel
        p.init_images[0] = p.init_images[0].convert("RGB")

        if use_deepbooru:
            interrogator = deepbooru.model.tag(p.init_images[0])
        else:
            interrogator = shared.interrogator.interrogate(p.init_images[0])

        if use_weight:
            if p.prompt == "":
                p.prompt = Script.interrogator
            elif in_front:
                p.prompt = f"{p.prompt}, ({interrogator}:{prompt_weight})"
            else:
                p.prompt = f"({interrogator}:{prompt_weight}), {p.prompt}"
        else:
            if p.prompt == "":
                p.prompt = Script.interrogator
            elif in_front:
                p.prompt = f"{p.prompt}, {interrogator}"
            else:
                p.prompt = f"{interrogator}, {p.prompt}"

        print(f"Prompt: {p.prompt}")

        processed = process_images(p)

        # Restore the UI elements we modified
        p.prompt = raw_prompt

        return processed
