import gradio as gr
from modules import scripts, shared, deepbooru
from modules.processing import process_images


class Script(scripts.Script):
    
    # :|
    orginal_prompt = "Some"
    interrogator = "Ramdom text"
    
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
        
        # :|
        if in_front:
            _check = f"{Script.orginal_prompt}, ({Script.interrogator}:{prompt_weight})"
        else:
            _check = f"({Script.interrogator}:{prompt_weight}), {Script.orginal_prompt}"
        if p.prompt != _check:
            Script.orginal_prompt = p.prompt
        

        if use_deepbooru:
            prompt = deepbooru.model.tag(p.init_images[0])
        else:
            prompt = shared.interrogator.interrogate(p.init_images[0])
        Script.interrogator = prompt


        p.prompt = ""
        if Script.orginal_prompt in ["Some", ""]:
            p.prompt = Script.interrogator
        elif in_front:
            p.prompt = f"{Script.orginal_prompt}, ({Script.interrogator}:{prompt_weight})"
        else:
            p.prompt = f"({Script.interrogator}:{prompt_weight}), {Script.orginal_prompt}"

        print(f"Prompt: {p.prompt}")
        return process_images(p)
