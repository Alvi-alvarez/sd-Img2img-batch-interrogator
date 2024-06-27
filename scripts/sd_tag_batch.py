import gradio as gr
import re
from modules import scripts, deepbooru
from modules.processing import process_images
import modules.shared as shared
import os
import requests
from io import BytesIO
import base64
from modules import script_callbacks

"""

Thanks to Mathias Russ.
Thanks to Smirking Kitsune.

"""

class Script(scripts.Script):
    server_address = None

    @classmethod
    def set_server_address(cls, demo, app, *args, **kwargs):
        cls.server_address = demo.local_url
        print(f"Server address set to: {cls.server_address}")
    
    @classmethod
    def get_server_address(cls):
        if cls.server_address:
            return cls.server_address
        
        # Fallback to the brute force method if server_address is not set
        # Initial testing indicates that fallback method might not be needed...
        print("Server address not set. Falling back to brute force method.")
        ports = range(7860, 7960) # Gradio will increment port 100 times if default and subsequent desired ports are unavailable. 
        for port in ports:
            url = f"http://127.0.0.1:{port}/"
            try:
                response = requests.get(f"{url}internal/ping", timeout=5)
                if response.status_code == 200:
                    return url
            except requests.RequestException as error:
                print(f"API not available on port {port}: {error}")

        print("API not found on any port")
        return None
    
    def title(self):
        return "Img2img batch interrogator"

    def show(self, is_img2img):
        return is_img2img

    def b_clicked(o):
        return gr.Button.update(interactive=True)
    
    def is_interrogator_enabled(self, interrogator):
        api_address = f"{self.get_server_address()}sdapi/v1/extensions"
        headers = {'accept': 'application/json'}

        try:
            response = requests.get(api_address, headers=headers)
            response.raise_for_status()
            extensions = response.json()

            for extension in extensions:
                if extension['name'] == interrogator:
                    return extension['enabled']

            return False
        except requests.RequestException:
            print(f"Error occurred while fetching extension: {interrogator}")
            return False

    # Removes unsupported interrogators, support may vary depending on client
    def update_model_choices(self, current_choices):
        all_options = ["CLIP (API)", "CLIP (Native)", "Deepbooru (Native)", "WD (API)"]
    
        if not self.is_interrogator_enabled('clip-interrogator-ext'):
            all_options.remove("CLIP (API)")
        if not self.is_interrogator_enabled('stable-diffusion-webui-wd14-tagger'):
            all_options.remove("WD (API)")
    
        # Keep the current selections if they're still valid
        updated_choices = [choice for choice in current_choices if choice in all_options]
    
        return gr.Dropdown.update(choices=all_options, value=updated_choices)
    
    # Function to load CLIP models
    def load_clip_models(self):
        models = self.get_clip_API_models()
        return gr.Dropdown.update(choices=models if models else None)
        
    # Function to load WD models
    def load_wd_models(self):
        models = self.get_WD_API_models()
        return gr.Dropdown.update(choices=models if models else None)
    
    def ui(self, is_img2img):
        model_options = ["CLIP (API)", "CLIP (Native)", "Deepbooru (Native)", "WD (API)"]
        model_selection = gr.Dropdown(choices=model_options, label="Select Interrogation Model(s)", multiselect=True, value="Deepbooru (Native)")
        
        in_front = gr.Radio(
            choices=["Prepend to prompt", "Append to prompt"],
            value="Prepend to prompt",
            label="Interrogator result position"
        )
        
        def update_prompt_weight_visibility(use_weight):
            return gr.Slider.update(visible=use_weight)
            
        use_weight = gr.Checkbox(label="Use Interrogator Prompt Weight", value=True)
        prompt_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Interrogator Prompt Weight", visible=True)
        
        # CLIP API Options
        def update_clip_api_visibility(model_selection):
            is_visible = "CLIP (API)" in model_selection
            if is_visible:
                clip_models = self.load_clip_models()
                return gr.Accordion.update(visible=True), clip_models
            else:
                return gr.Accordion.update(visible=False), gr.Dropdown.update()
        
        clip_api_accordion = gr.Accordion("CLIP API Options:", open=False, visible=False)
        with clip_api_accordion:
            clip_api_model = gr.Dropdown(choices=[], value='ViT-L-14/openai', label="CLIP API Model")
            clip_api_mode = gr.Radio(choices=["fast", "best", "classic", "negative"], label="CLIP API Mode", value="fast")

        # WD API Options
        def update_wd_api_visibility(model_selection):
            is_visible = "WD (API)" in model_selection
            if is_visible:
                wd_models = self.load_wd_models()
                return gr.Accordion.update(visible=True), wd_models
            else:
                return gr.Accordion.update(visible=False), gr.Dropdown.update()

        wd_api_accordion = gr.Accordion("WD API Options:", open=False, visible=False)
        with wd_api_accordion:
            wd_api_model = gr.Dropdown(choices=[], value='wd-v1-4-moat-tagger.v2', label="WD API Model")
            wd_underscore_fix = gr.Checkbox(label="Remove Underscores from Tags", value=True)
            wd_threshold = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Threshold")
            unload_wd_models_afterwords = gr.Checkbox(label="Unload WD Model After Use", value=True)
            unload_wd_models_button = gr.Button(value="Unload WD Models")

        # Function to load custom filter from file
        def load_custom_filter(custom_filter):
            with open("extensions/sd-Img2img-batch-interrogator/custom_filter.txt", "r") as file:
                custom_filter = file.read()
                return custom_filter

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
        model_selection.change(fn=update_clip_api_visibility, inputs=[model_selection], outputs=[clip_api_accordion, clip_api_model])
        model_selection.change(fn=update_wd_api_visibility, inputs=[model_selection], outputs=[wd_api_accordion, wd_api_model])
        load_custom_filter_button.click(load_custom_filter, inputs=custom_filter, outputs=custom_filter)
        unload_wd_models_button.click(self.post_wd_api_unload, inputs=None, outputs=None)
        use_weight.change(fn=update_prompt_weight_visibility, inputs=[use_weight], outputs=[prompt_weight])


        return [in_front, prompt_weight, model_selection, use_weight, no_duplicates, use_negatives, use_custom_filter, custom_filter, clip_api_model, clip_api_mode, wd_api_model, wd_threshold, wd_underscore_fix, unload_wd_models_afterwords]
        
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

    def run(self, p, in_front, prompt_weight, model_selection, use_weight, no_duplicates, use_negatives, use_custom_filter, custom_filter, clip_api_model, clip_api_mode, wd_api_model, wd_threshold, wd_underscore_fix, unload_wd_models_afterwords):
        raw_prompt = p.prompt
        interrogator = ""

        # fix alpha channel
        p.init_images[0] = p.init_images[0].convert("RGB")
        
        first = True  # Two interrogator concatenation correction boolean
        for model in model_selection:
            # This prevents two interrogators from being incorrectly concatenated
            if first == False:
                interrogator += ", "
            first = False
            # Should add the interrogators in the order determined by the model_selection list
            if model == "Deepbooru (Native)":
                interrogator += deepbooru.model.tag(p.init_images[0])
            elif model == "CLIP (Native)":
                interrogator += shared.interrogator.interrogate(p.init_images[0])
            elif model == "CLIP (API)":
                interrogator += self.post_clip_api_prompt(p.init_images[0], clip_api_model, clip_api_mode)
            elif model == "WD (API)":
                interrogator += self.post_wd_api_tagger(p.init_images[0], wd_api_model, wd_threshold, wd_underscore_fix)
       
        # Remove duplicate prompt content from interrogator prompt
        if no_duplicates:
            interrogator = self.filter_words(interrogator, raw_prompt)
        # Remove negative prompt content from interrogator prompt
        if use_negatives:
            interrogator = self.filter_words(interrogator, p.negative_prompt)
        # Remove custom prompt content from interrogator prompt
        if use_custom_filter:
            interrogator = self.filter_words(interrogator, custom_filter)
            # Save custom filter to text file
            with open("extensions/sd-Img2img-batch-interrogator/custom_filter.txt", "w") as file:
                file.write(custom_filter)

        if use_weight:
            if p.prompt == "":
                p.prompt = interrogator
            elif in_front == "Append to prompt":
                p.prompt = f"{p.prompt}, ({interrogator}:{prompt_weight})"
            else:
                p.prompt = f"({interrogator}:{prompt_weight}), {p.prompt}"
        else:
            if p.prompt == "":
                p.prompt = interrogator
            elif in_front == "Append to prompt":
                p.prompt = f"{p.prompt}, {interrogator}"
            else:
                p.prompt = f"{interrogator}, {p.prompt}"
        
        if unload_wd_models_afterwords and "WD (API)" in model_selection:
            self.post_wd_api_unload()
        
        print(f"Prompt: {p.prompt}")

        processed = process_images(p)

        # Restore the UI elements we modified
        p.prompt = raw_prompt

        return processed


    # CLIP API Model Identification
    def get_clip_API_models(self):
        # Ensure CLIP Interrogator is present and accessible
        try:
            api_address = f"{self.get_server_address()}interrogator/models"
            response = requests.get(api_address)
            response.raise_for_status()
            models = response.json()
            if not models:
                raise Exception("No CLIP Interrogator models found.")
        except Exception as error:
            print(f"Error accessing CLIP Interrogator API: {error}")
            return []
        return models
        
    # WD API Model Identification
    def get_WD_API_models(self):
        # Ensure WD Interrogator is present and accessible
        try:
            api_address = f"{self.get_server_address()}tagger/v1/interrogators"
            response = requests.get(api_address)
            response.raise_for_status()
            models = response.json()["models"]
            if not models:
                raise Exception("No WD Tagger models found.")
        except Exception as error:
            print(f"Error accessing WD Tagger API: {error}")
            return []
        return models
         
    # CLIP API Prompt Generator 
    def post_clip_api_prompt(self, image, model_name, mode):
        print("Starting CLIP Interrogator API interaction...")
        # Ensure the model and mode are provided
        if not model_name:
            print("CLIP API model is required.")
            return ""

        # Encode the image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Get the prompt from the CLIP API
        try:
            payload = {
                "image": img_str,
                "mode": mode,
                "clip_model_name": model_name
            }
            api_address = f"{self.get_server_address()}interrogator/prompt"
            response = requests.post(api_address, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("prompt", "")
        except Exception as error:
            print(f"Error generating prompt with CLIP API: {error}")
            return ""
            
    # WD API Interrogation Tagger 
    def post_wd_api_tagger(self, image, model_name, threshold, underscore):
        print("Starting WD Tagger API interaction...")
        # Ensure the model and mode are provided
        if not model_name:
            print("WD API model is required.")
            return ""

        # Encode the image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Get the prompt from the WD API
        try:
            payload = {
                "image": img_str,
                "model": model_name,
                "threshold": threshold,
                "queue": "",
                "name_in_queue": ""
            }
            api_address = f"{self.get_server_address()}tagger/v1/interrogate"
            # WARNING: Removing `timeout` could result in a frozen client if the queue_lock is locked. If you need more time add more time, do not remove or risk DEADLOCK.
            # Note: If WD Tagger did not load a model, it is likely that WD Tagger specifically queue_lock (FIFOLock) is concerned with your system's threading and thinks running could cause process starvation...
            response = requests.post(api_address, json=payload, timeout=120) 
            response.raise_for_status()
            result = response.json()
            
            tags_list = result.get("caption", {}).get("tag", [])
        
            if underscore:
                tags_spaced = [tag.replace('_', ' ') for tag in tags_list]
                tags_string = ", ".join(tags_spaced)
            else:
                tags_string = ", ".join(tags_list)

            return tags_string
        except Exception as error:
            print(f"Error generating prompt with WD API: {error}")
            return ""        
            
    # WD API Model Unloader 
    def post_wd_api_unload(self):
        try:
            api_address = f"{self.get_server_address()}tagger/v1/unload-interrogators"
            response = requests.post(api_address, json='') 
            response.raise_for_status()
            
        except Exception as error:
            print(f"Error Unloading models with WD API: {error}")   

script_callbacks.on_app_started(Script.set_server_address)
