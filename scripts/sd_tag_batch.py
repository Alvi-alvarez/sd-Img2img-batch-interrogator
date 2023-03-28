import gradio as gr
from modules import scripts, devices, lowvram, shared
from clip_interrogator import Config, Interrogator
from modules.processing import process_images

ci = None


def unload():
    global ci
    if ci is not None:
        print("Offloading CLIP Interrogator...")
        ci.caption_model = ci.caption_model.to(devices.cpu)
        ci.clip_model = ci.clip_model.to(devices.cpu)
        ci.caption_offloaded = True
        ci.clip_offloaded = True
        devices.torch_gc()


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
        mode = gr.Dropdown(["classic", "fast"], label="mode", value="classic")
        btn = gr.Button(value="unload models")
        btn.click(unload)
        return [in_front, mode, prompt_weight]

    def run(self, p, in_front, mode, prompt_weight):
        global ci
        if ci is None:
            config = Config(
                device=devices.get_optimal_device(),
                cache_path="models/clip-interrogator",
                clip_model_name="ViT-L-14/openai",
                caption_model_name="blip-base",
            )
            ci = Interrogator(config)
        try:
            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()
                devices.torch_gc()
            if mode == "classic":
                prompt = ci.interrogate_classic(p.init_images[0])
            elif mode == "fast":
                prompt = ci.interrogate_fast(p.init_images[0])
            if in_front:
                p.prompt = f"{p.prompt}, ({prompt}:{prompt_weight})"
            else:
                p.prompt = f"({prompt}:{prompt_weight}), {p.prompt}"
            print(prompt)
        except RuntimeError as e:
            print(e)
        return process_images(p)
