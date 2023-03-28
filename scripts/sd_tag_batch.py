import gradio as gr
from modules import scripts, shared, deepbooru
from modules.processing import process_images


class Script(scripts.Script):
    def title(self):
        return "Img2img batch interrogator"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        in_front = gr.Checkbox(label="Prompt in front")
        prompt_weight = gr.Slider(
            0.0, 1.0, value=0.5, step=0.1, label="interrogator weight"
        )
        use_deepbooru = gr.Checkbox(label="Use deepbooru")
        return [in_front, prompt_weight, use_deepbooru]

    def run(self, p, in_front, prompt_weight, use_deepbooru):
        prompt = ""
        if use_deepbooru:
            prompt = deepbooru.model.tag(p.init_images[0])
        else:
            prompt = shared.interrogator.interrogate(p.init_images[0])
        if in_front:
            p.prompt = f"{p.prompt}, ({prompt}:{prompt_weight})"
        else:
            p.prompt = f"({prompt}:{prompt_weight}), {p.prompt}"
        print(f"Prompt: {p.prompt}")
        return process_images(p)
