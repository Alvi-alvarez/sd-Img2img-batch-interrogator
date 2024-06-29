import gradio as gr
import re
from modules import scripts, deepbooru, script_callbacks, shared
from modules.processing import process_images
import sys
import importlib.util

NAME = "Img2img batch interrogator"

"""

Thanks to Mathias Russ.
Thanks to Smirking Kitsune.

"""

# Extention List Crawler
def get_extensions_list():
    from modules import extensions
    extensions.list_extensions()
    ext_list = []
    for ext in extensions.extensions:
        ext: extensions.Extension
        ext.read_info_from_repo()
        if ext.remote is not None:
            ext_list.append({
                "name": ext.name,
                "enabled":ext.enabled
            })
    return ext_list

# Extention Checker
def is_interrogator_enabled(interrogator):
    for ext in get_extensions_list():
        if ext["name"] == interrogator:
            return ext["enabled"]
    return False

# EXT Importer
def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

class Script(scripts.Script):
    wd_ext_utils = None
    clip_ext = None

    # Checks for CLIP EXT to see if it is installed and enabled
    @classmethod
    def load_clip_ext_module(cls):
        if is_interrogator_enabled('clip-interrogator-ext'):
            cls.clip_ext = import_module("clip-interrogator-ext", "extensions/clip-interrogator-ext/scripts/clip_interrogator_ext.py")
            return cls.clip_ext
        return None

    # Checks for WD EXT to see if it is installed and enabled
    @classmethod
    def load_wd_ext_module(cls):
        if is_interrogator_enabled('stable-diffusion-webui-wd14-tagger'):
            sys.path.append('extensions/stable-diffusion-webui-wd14-tagger')
            cls.wd_ext_utils = import_module("utils", "extensions/stable-diffusion-webui-wd14-tagger/tagger/utils.py")
            return cls.wd_ext_utils
        return None
    
    # Initiates extenion check at startup for CLIP EXT
    @classmethod
    def load_clip_ext_module_wrapper(cls, *args, **kwargs):
        return cls.load_clip_ext_module()

    # Initiates extenion check at startup for WD EXT
    @classmethod
    def load_wd_ext_module_wrapper(cls, *args, **kwargs):
        return cls.load_wd_ext_module()
    
    def title(self):
        return NAME

    def show(self, is_img2img):
        # return scripts.AlwaysVisible if is_img2img else False
        return is_img2img

    def b_clicked(o):
        return gr.Button.update(interactive=True)
    
    # Removes unsupported interrogators, support may vary depending on client
    def update_model_choices(self, current_choices):
        all_options = ["CLIP (EXT)", "CLIP (Native)", "Deepbooru (Native)", "WD (EXT)"]

        if not is_interrogator_enabled('clip-interrogator-ext'):
            all_options.remove("CLIP (EXT)")
        if not is_interrogator_enabled('stable-diffusion-webui-wd14-tagger'):
            all_options.remove("WD (EXT)")

        updated_choices = [choice for choice in current_choices if choice in all_options]

        return gr.Dropdown.update(choices=all_options, value=updated_choices)
    
    # Function to load CLIP models list into CLIP model selector
    def load_clip_models(self):
        if self.clip_ext is not None:
            models = self.clip_ext.get_models()
            return gr.Dropdown.update(choices=models if models else None)
        return gr.Dropdown.update(choices=None)
        
    # Function to load WD models list into WD model selector
    def load_wd_models(self):
        if self.wd_ext_utils is not None:
            models = self.get_WD_EXT_models()
            return gr.Dropdown.update(choices=models if models else None)
        return gr.Dropdown.update(choices=None)

    # Gets a list of WD models from WD EXT
    def get_WD_EXT_models(self):
        if self.wd_ext_utils is not None:
            try:
                self.wd_ext_utils.refresh_interrogators()
                models = list(self.wd_ext_utils.interrogators.keys())
                if not models:
                    raise Exception("No WD Tagger models found.")
                return models
            except Exception as error:
                print(f"Error accessing WD Tagger: {error}")
        return []

    #Unloads CLIP Models
    def unload_clip_models(self):
        if self.clip_ext is not None:
            self.clip_ext.unload()

    #Unloads WD Models
    def unload_wd_models(self):
        unloaded_models = 0
        if self.wd_ext_utils is not None:
            for interrogator in self.wd_ext_utils.interrogators.values():
                if interrogator.unload(): 
                    unloaded_models = unloaded_models + 1
            print(f"Unloaded {unloaded_models} Tagger Model(s).")
                

    
    # depending on if CLIP (EXT) is present, CLIP (EXT) could be removed from model selector
    def update_clip_ext_visibility(self, model_selection):
        is_visible = "CLIP (EXT)" in model_selection
        if is_visible:
            clip_models = self.load_clip_models()
            return gr.Accordion.update(visible=True), clip_models
        else:
            return gr.Accordion.update(visible=False), gr.Dropdown.update()
    
    # depending on if WD (EXT) is present, WD (EXT) could be removed from model selector
    def update_wd_ext_visibility(self, model_selection):
        is_visible = "WD (EXT)" in model_selection
        if is_visible:
            wd_models = self.load_wd_models()
            return gr.Accordion.update(visible=True), wd_models
        else:
            return gr.Accordion.update(visible=False), gr.Dropdown.update()
    
    # Depending on if prompt weight is enabled the slider will be dynamically visible
    def update_prompt_weight_visibility(self, use_weight):
        return gr.Slider.update(visible=use_weight)
    
    # Function to load custom filter from file
    def load_custom_filter(self, custom_filter):
        with open("extensions/sd-Img2img-batch-interrogator/custom_filter.txt", "r") as file:
            custom_filter = file.read()
            return custom_filter
    
    def ui(self, is_img2img):
        with gr.Group():
            model_options = ["CLIP (EXT)", "CLIP (Native)", "Deepbooru (Native)", "WD (EXT)"]
            model_selection = gr.Dropdown(choices=model_options, 
                label="Interrogation Model(s):",
                multiselect=True, 
                value=None,
            )
            
            in_front = gr.Radio(
                choices=["Prepend to prompt", "Append to prompt"], 
                value="Prepend to prompt", 
                label="Interrogator result position")

            use_weight = gr.Checkbox(label="Use Interrogator Prompt Weight", value=True)
            prompt_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Interrogator Prompt Weight", visible=True)
            
            # CLIP EXT Options
            clip_ext_accordion = gr.Accordion("CLIP EXT Options:", open=False, visible=False)
            with clip_ext_accordion:
                clip_ext_model = gr.Dropdown(choices=[], value='ViT-L-14/openai', label="CLIP Extension Model(s):", multiselect=True)
                clip_ext_mode = gr.Radio(choices=["best", "fast", "classic", "negative"], value='best', label="CLIP Extension Mode")
                unload_clip_models_afterwords = gr.Checkbox(label="Unload CLIP Interrogator After Use", value=True)
                unload_clip_models_button = gr.Button(value="Unload All CLIP Interrogators")
                
            # WD EXT Options
            wd_ext_accordion = gr.Accordion("WD EXT Options:", open=False, visible=False)
            with wd_ext_accordion:
                wd_ext_model = gr.Dropdown(choices=[], value='wd-v1-4-moat-tagger.v2', label="WD Extension Model(s):", multiselect=True)
                wd_threshold = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Threshold")
                wd_underscore_fix = gr.Checkbox(label="Remove Underscores from Tags", value=True)
                unload_wd_models_afterwords = gr.Checkbox(label="Unload Tagger After Use", value=True)
                unload_wd_models_button = gr.Button(value="Unload All Tagger Models")
                    
            with gr.Accordion("Filtering tools:"):
                no_duplicates = gr.Checkbox(label="Filter Duplicate Prompt Content from Interrogation", value=False)
                use_negatives = gr.Checkbox(label="Filter Negative Prompt Content from Interrogation", value=False)
                use_custom_filter = gr.Checkbox(label="Filter Custom Prompt Content from Interrogation", value=False)
                custom_filter = gr.Textbox(
                    label="Custom Filter Prompt",
                    placeholder="Prompt content separated by commas. Warning ignores attention syntax, parentheses '()' and colon suffix ':XX.XX' are discarded.",
                    show_copy_button=True
                )
                # Button to load custom filter from file
                load_custom_filter_button = gr.Button(value="Load Last Custom Filter")
                
            # Listeners
            model_selection.select(fn=self.update_model_choices, inputs=[model_selection], outputs=[model_selection])
            model_selection.change(fn=self.update_clip_ext_visibility, inputs=[model_selection], outputs=[clip_ext_accordion, clip_ext_model])
            model_selection.change(fn=self.update_wd_ext_visibility, inputs=[model_selection], outputs=[wd_ext_accordion, wd_ext_model])
            load_custom_filter_button.click(self.load_custom_filter, inputs=custom_filter, outputs=custom_filter)
            unload_clip_models_button.click(self.unload_clip_models, inputs=None, outputs=None)
            unload_wd_models_button.click(self.unload_wd_models, inputs=None, outputs=None)
            use_weight.change(fn=self.update_prompt_weight_visibility, inputs=[use_weight], outputs=[prompt_weight])
            
            
        return [in_front, prompt_weight, model_selection, use_weight, no_duplicates, use_negatives, use_custom_filter, custom_filter, clip_ext_model, clip_ext_mode, wd_ext_model, wd_threshold, wd_underscore_fix, unload_clip_models_afterwords, unload_wd_models_afterwords]

    # Required to parse information from a string that is between () or has :##.## suffix
    def remove_attention(self, words):
        # Define a regular expression pattern to match attention-related suffixes
        pattern = r":\d+(\.\d+)?"
        # Remove attention-related suffixes using regex substitution
        words = re.sub(pattern, "", words)
        
        # Replace escaped left parenthesis with temporary placeholder
        words = re.sub(r"\\\(", r"TEMP_LEFT_PLACEHOLDER", words)
        # Replace escaped right parenthesis with temporary placeholder
        words = re.sub(r"\\\)", r"TEMP_RIGHT_PLACEHOLDER", words)
        # Define a regular expression pattern to match parentheses and their content
        pattern = r"(\(|\))"
        # Remove parentheses using regex substitution
        words = re.sub(pattern, "", words)
        # Restore escaped left parenthesis
        words = re.sub(r"TEMP_LEFT_PLACEHOLDER", r"\\(", words)
        # Restore escaped right parenthesis
        words = re.sub(r"TEMP_RIGHT_PLACEHOLDER", r"\\)", words)
        
        return words.strip()

    # Tag filtering, removes negative tags from prompt
    def filter_words(self, prompt, negative):
        # Corrects a potential error where negative is nonetype
        if negative is None:
            negative = ""
        
        # Split prompt and negative strings into lists of words
        prompt_words = [word.strip() for word in prompt.split(",")]
        negative_words = [self.remove_attention(word.strip()) for word in negative.split(",")]
        
        # Filter out words from prompt that are in negative
        filtered_words = [word for word in prompt_words if self.remove_attention(word) not in negative_words]
        
        # Join filtered words back into a string
        filtered_prompt = ", ".join(filtered_words)
        
        return filtered_prompt

    # For WD Tagger, removes underscores from tags that should have spaces
    def replace_underscores(self, tag):
        preserve_patterns = [
            "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", 
            "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"
        ]
        if tag in preserve_patterns:
            return tag
        return tag.replace('_', ' ')

    def run(self, p, in_front, prompt_weight, model_selection, use_weight, no_duplicates, use_negatives, use_custom_filter, custom_filter, clip_ext_model, clip_ext_mode, wd_ext_model, wd_threshold, wd_underscore_fix, unload_clip_models_afterwords, unload_wd_models_afterwords):
        raw_prompt = p.prompt
        interrogation = ""
        preliminary_interrogation = ""
        
        # fix alpha channel
        p.init_images[0] = p.init_images[0].convert("RGB")
        
        for model in model_selection:
            # Should add the interrogators in the order determined by the model_selection list
            if model == "Deepbooru (Native)":
                preliminary_interrogation = deepbooru.model.tag(p.init_images[0]) + ", "
            elif model == "CLIP (Native)":
                preliminary_interrogation = shared.interrogator.interrogate(p.init_images[0]) + ", "
            elif model == "CLIP (EXT)":
                if self.clip_ext is not None:
                    for clip_model in clip_ext_model:
                        preliminary_interrogation = self.clip_ext.image_to_prompt(p.init_images[0], clip_ext_mode, clip_model) + ", "
                    if unload_clip_models_afterwords:
                        self.clip_ext.unload()
            elif model == "WD (EXT)":
                if self.wd_ext_utils is not None:
                    for wd_model in wd_ext_model:
                        rating, tags = self.wd_ext_utils.interrogators[wd_model].interrogate(p.init_images[0])
                        tags_list = [tag for tag, conf in tags.items() if conf > wd_threshold]
                        if wd_underscore_fix:
                            tags_spaced = [self.replace_underscores(tag) for tag in tags_list]
                            preliminary_interrogation = ", ".join(tags_spaced) + ", "
                        else:
                            preliminary_interrogation += ", ".join(tags_list) + ", "
                        if unload_wd_models_afterwords:
                            self.wd_ext_utils.interrogators[wd_model].unload()
            
            # Filter prevents overexaggeration of tags due to interrogation models having similar results 
            interrogation += self.filter_words(preliminary_interrogation, interrogation)
        
        # Remove duplicate prompt content from interrogator prompt
        if no_duplicates:
            interrogation = self.filter_words(interrogation, raw_prompt)
        # Remove negative prompt content from interrogator prompt
        if use_negatives:
            interrogation = self.filter_words(interrogation, p.negative_prompt)
        # Remove custom prompt content from interrogator prompt
        if use_custom_filter:
            interrogation = self.filter_words(interrogation, custom_filter)
            # Save custom filter to text file
            with open("extensions/sd-Img2img-batch-interrogator/custom_filter.txt", "w") as file:
                file.write(custom_filter)
        
        if use_weight:
            if p.prompt == "":
                p.prompt = interrogation
            elif in_front == "Append to prompt":
                p.prompt = f"{p.prompt}, ({interrogation}:{prompt_weight})"
            else:
                p.prompt = f"({interrogation}:{prompt_weight}), {p.prompt}"
        else:
            if p.prompt == "":
                p.prompt = interrogation
            elif in_front == "Append to prompt":
                p.prompt = f"{p.prompt}, {interrogation}"
            else:
                p.prompt = f"{interrogation}, {p.prompt}"
        
        print(f"Prompt: {p.prompt}")
        
        processed = process_images(p)
        
        # Restore the UI elements we modified
        p.prompt = raw_prompt
        
        return processed

#Startup Callbacks
script_callbacks.on_app_started(Script.load_clip_ext_module_wrapper)
script_callbacks.on_app_started(Script.load_wd_ext_module_wrapper)
