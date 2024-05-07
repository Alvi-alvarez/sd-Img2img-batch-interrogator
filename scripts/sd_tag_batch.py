import gradio as gr
import unicodedata
from modules import scripts, deepbooru
from modules.processing import process_images
import modules.shared as shared


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join(c for c in nfkd_form if not unicodedata.combining(c))


class Script(scripts.Script):
    # :|
    original_prompt = "Some"
    interrogator = "Random text"

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

        # :|
        if in_front:
            _check = f"{Script.original_prompt}, ({Script.interrogator}:{prompt_weight})"
        else:
            _check = f"({Script.interrogator}:{prompt_weight}), {Script.original_prompt}"

        if p.prompt not in [_check, Script.interrogator]:
            Script.original_prompt = remove_accents(p.prompt)

        # fix alpha channel
        p.init_images[0] = p.init_images[0].convert('RGB')

        if use_deepbooru:
            prompt = deepbooru.model.tag(p.init_images[0])
        else:
            prompt = shared.interrogator.interrogate(p.init_images[0])
        Script.interrogator = prompt

        p.prompt = ""
        if use_weight:
            if Script.original_prompt in ["Some", ""]:
                p.prompt = Script.interrogator
            elif in_front:
                p.prompt = f"{Script.original_prompt}, ({Script.interrogator}:{prompt_weight})"
            else:
                p.prompt = f"({Script.interrogator}:{prompt_weight}), {Script.original_prompt}"
        else:
            if Script.original_prompt in ["Some", ""]:
                p.prompt = Script.interrogator
            elif in_front:
                p.prompt = f"{Script.original_prompt}, {Script.interrogator}"
            else:
                p.prompt = f"{Script.interrogator}, {Script.original_prompt}"

        print(f"Prompt: {p.prompt}")
        return process_images(p)
