import gradio as gr
import re
from modules import scripts, deepbooru, script_callbacks, shared
from modules.processing import process_images
from modules.shared import state
import sys
import importlib.util

NAME = "Img2img Batch Interrogator"

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

class Script(scripts.ScriptBuiltinUI):
    wd_ext_utils = None
    clip_ext = None
    first = True
    prompt_contamination = ""

    def title(self):
        # "Img2img Batch Interrogator"
        return NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False
 
    # Checks for CLIP EXT to see if it is installed and enabled
    @classmethod
    def load_clip_ext_module(cls):
        if is_interrogator_enabled('clip-interrogator-ext'):
            cls.clip_ext = import_module("clip-interrogator-ext", "extensions/clip-interrogator-ext/scripts/clip_interrogator_ext.py")
            print(f"[{NAME} LOADER]: `clip-interrogator-ext` found...")
            return cls.clip_ext
        print(f"[{NAME} LOADER]: `clip-interrogator-ext` NOT found!")
        return None

    # Initiates extenion check at startup for CLIP EXT
    @classmethod
    def load_clip_ext_module_wrapper(cls, *args, **kwargs):
        return cls.load_clip_ext_module()

    # Checks for WD EXT to see if it is installed and enabled
    @classmethod
    def load_wd_ext_module(cls):
        if is_interrogator_enabled('stable-diffusion-webui-wd14-tagger'):
            sys.path.append('extensions/stable-diffusion-webui-wd14-tagger')
            cls.wd_ext_utils = import_module("utils", "extensions/stable-diffusion-webui-wd14-tagger/tagger/utils.py")
            print(f"[{NAME} LOADER]: `stable-diffusion-webui-wd14-tagger` found...")
            return cls.wd_ext_utils
        print(f"[{NAME} LOADER]: `stable-diffusion-webui-wd14-tagger` NOT found!")
        return None
    
    # Initiates extenion check at startup for WD EXT
    @classmethod
    def load_wd_ext_module_wrapper(cls, *args, **kwargs):
        return cls.load_wd_ext_module()
        
    # Initiates prompt reset on image save
    @classmethod
    def load_custom_filter_module_wrapper(cls, *args, **kwargs):
        return cls.load_custom_filter()
    
    # Button interaction handler
    def b_clicked(o):
        return gr.Button.update(interactive=True)
    
    #Experimental Tool, prints dev statements to the console
    def debug_print(self, debug_mode, message):
        if debug_mode:
            print(f"[{NAME} DEBUG]: {message}")
    
    # Function to clean the custom_filter
    def clean_string(self, input_string):
        # Split the string into a list
        items = input_string.split(',')
        # Clean up each item: strip whitespace and convert to lowercase
        cleaned_items = [item.strip() for item in items if item.strip()]
        # Remove duplicates while preserving order
        unique_items = []
        seen = set()
        for item in cleaned_items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        # Join the cleaned, unique items back into a string
        return ', '.join(unique_items)

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

    # Initial Model Options generator, only add supported interrogators, support may vary depending on client
    def get_initial_model_options(self):
        options = ["CLIP (Native)", "Deepbooru (Native)"]
        if is_interrogator_enabled('clip-interrogator-ext'):
            options.insert(0, "CLIP (EXT)")
        if is_interrogator_enabled('stable-diffusion-webui-wd14-tagger'):
            options.append("WD (EXT)")
        return options
        
    # Gets a list of WD models from WD EXT
    def get_WD_EXT_models(self):
        if self.wd_ext_utils is not None:
            try:
                self.wd_ext_utils.refresh_interrogators()
                models = list(self.wd_ext_utils.interrogators.keys())
                if not models:
                    raise Exception("[{NAME} DEBUG]: No WD Tagger models found.")
                return models
            except Exception as errorrror:
                print(f"[{NAME} ERROR]: Error accessing WD Tagger: {error}")
        return []
    
    # Function to load CLIP models list into CLIP model selector
    def load_clip_models(self):
        if self.clip_ext is not None:
            models = self.clip_ext.get_models()
            return gr.Dropdown.update(choices=models if models else None)
        return gr.Dropdown.update(choices=None)
        
    # Function to load custom filter from file
    def load_custom_filter(self):
        try:
            with open("extensions/sd-Img2img-batch-interrogator/custom_filter.txt", "r", encoding="utf-8") as file:
                custom_filter = file.read()
                return custom_filter
        except Exception as error:
            print(f"[{NAME} ERROR]: Error loading custom filter: {error}")  
            return ""
        
    # Function to load WD models list into WD model selector
    def load_wd_models(self):
        if self.wd_ext_utils is not None:
            models = self.get_WD_EXT_models()
            return gr.Dropdown.update(choices=models if models else None)
        return gr.Dropdown.update(choices=None)
    
    # Refresh the model_selection dropdown
    def refresh_model_options(self):
        new_options = self.get_initial_model_options()
        return gr.Dropdown.update(choices=new_options)
    
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
    
    # Experimental Tool, removes puncutation, but tries to keep a variety of known emojis
    def remove_punctuation(self, text):
        # List of text emojis to preserve
        skipables = ["'s", "...", ":-)", ":)", ":-]", ":]", ":->", ":>", "8-)", "8)", ":-}", ":}", ":^)", "=]", "=)", ":-D", ":D", "8-D", "8D", "=D", "=3", "B^D", 
            "c:", "C:", "x-D", "X-D", ":-))", ":))", ":-(", ":(", ":-c", ":c", ":-<", ":<", ":-[", ":[", ":-||", ":{", ":@", ":(", ";(", ":'-(", ":'(", ":=(", ":'-)", 
            ":')", ">:(", ">:[", "D-':", "D:<", "D:", "D;", "D=", ":-O", ":O", ":-o", ":o", ":-0", ":0", "8-0", ">:O", "=O", "=o", "=0", ":-3", ":3", "=3", ">:3", 
            ":-*", ":*", ":x", ";-)", ";)", "*-)", "*)", ";-]", ";]", ";^)", ";>", ":-,", ";D", ";3", ":-P", ":P", "X-P", "x-p", ":-p", ":p", ":-Þ", ":Þ", ":-þ", 
            ":þ", ":-b", ":b", "d:", "=p", ">:P", ":-/", ":/", ":-.", ">:/", "=/", ":L", "=L", ":S", ":-|", ":|", ":$", "://)", "://3", ":-X", ":X", ":-#", ":#", 
            ":-&", ":&", "O:-)", "O:)", "0:-3", "0:3", "0:-)", "0:)", "0;^)", ">:-)", ">:)", "}:-)", "}:)", "3:-)", "3:)", ">;-)", ">;)", ">:3", ">;3", "|;-)", "|-O", 
            "B-)", ":-J", "#-)", "%-)", "%)", ":-###..", ":###..", "<:-|", "',:-|", "',:-l", ":E", "8-X", "8=X", "x-3", "x=3", "~:>", "@};-", "@}->--", "@}-;-'---", 
            "@>-->--", "8====D", "8===D", "8=D", "3=D", "8=>", "8===D~~~", "*<|:-)", "</3", "<\3", "<3", "><>", "<><", "<*)))-{", "><(((*>", "\o/", "*\0/*", "o7", 
            "v.v", "._.", "._.;", "X_X", "x_x", "+_+", "X_x", "x_X", "<_<", ">_>", "<.<", ">.>", "O_O", "o_o", "O-O", "o-o", "O_o", "o_O", ">.<", ">_<", "^5", "o/\o", 
            ">_>^ ^<_<", "V.v.V"] # Maybe I should remove emojis with parenthesis () in them...
        # Temporarily replace text emojis with placeholders
        for i, noticables in enumerate(skipables):
            text = text.replace(noticables, f"SKIP_PLACEHOLDER_{i}")
        # Remove punctuation except commas
        text = re.sub(r'[^\w\s,]', '', text)
        # Split the text into tags
        tags = [tag.strip() for tag in text.split(',')]
        # Remove empty tags
        tags = [tag for tag in tags if tag]
        # Rejoin the tags
        text = ', '.join(tags)
        # Restore text emojis
        for i, noticables in enumerate(skipables):
            text = text.replace(f"SKIP_PLACEHOLDER_{i}", noticables)
        return text
    
    # For WD Tagger, removes underscores from tags that should have spaces
    def replace_underscores(self, tag):
        skipable = [
            "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", 
            "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"
        ]
        if tag in skipable:
            return tag
        return tag.replace('_', ' ')

    # Resets the prompt_contamination string, prompt_contamination is used to clean the p.prompt after it has been modified by a previous batch job
    def reset_prompt_contamination(self, debug_mode):
        """
        Note: prompt_contamination
            During the course of a process_batch, the p.prompt and p.all_prompts[0] 
            is going to become contaminated with previous interrogation in the batch, to 
            mitigate this problem, prompt_contamination is used to identify and remove contamination
        """
        self.debug_print(debug_mode, f"Reset was Called! The following prompt will be removed from the prompt_contamination cleaner: {self.prompt_contamination}")
        self.prompt_contamination = ""
    
    # Function to load custom filter from file
    def save_custom_filter(self, custom_filter):
        try:
            with open("extensions/sd-Img2img-batch-interrogator/custom_filter.txt", "w", encoding="utf-8") as file:
                file.write(custom_filter)
                print(f"[{NAME}]: Custom filter saved successfully.")
        except Exception as error:
            print(f"[{NAME} ERROR]: Error saving custom filter: {error}")  
        return self.update_save_confirmation_row_false()
    
    # depending on if CLIP (EXT) is present, CLIP (EXT) could be removed from model selector
    def update_clip_ext_visibility(self, model_selection):
        is_visible = "CLIP (EXT)" in model_selection
        if is_visible:
            clip_models = self.load_clip_models()
            return gr.Accordion.update(visible=True), clip_models
        else:
            return gr.Accordion.update(visible=False), gr.Dropdown.update()
    
    # Depending on if prompt weight is enabled the slider will be dynamically visible
    def update_prompt_weight_visibility(self, prompt_weight_mode):
        return gr.Slider.update(visible=prompt_weight_mode)
    
    def update_save_confirmation_row_false(self):
        return gr.Accordion.update(visible=False)
    
    def update_save_confirmation_row_true(self):
        return gr.Accordion.update(visible=True)
    
    # depending on if WD (EXT) is present, WD (EXT) could be removed from model selector
    def update_wd_ext_visibility(self, model_selection):
        is_visible = "WD (EXT)" in model_selection
        if is_visible:
            wd_models = self.load_wd_models()
            return gr.Accordion.update(visible=True), wd_models
        else:
            return gr.Accordion.update(visible=False), gr.Dropdown.update()
    
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
    
    def ui(self, is_img2img):
        tag_batch_ui = gr.Accordion(NAME, open=False)
        with tag_batch_ui:
            with gr.Row():
                model_selection = gr.Dropdown(
                    choices=self.get_initial_model_options(), 
                    label="Interrogation Model(s):",
                    multiselect=True
                )
                refresh_models_button = gr.Button("🔄", elem_classes="tool")
            
            in_front = gr.Radio(
                choices=["Prepend to prompt", "Append to prompt"], 
                value="Prepend to prompt", 
                label="Interrogator result position")
                        
            # CLIP EXT Options
            clip_ext_accordion = gr.Accordion("CLIP EXT Options:", visible=False)
            with clip_ext_accordion:
                clip_ext_model = gr.Dropdown(choices=[], value='ViT-L-14/openai', label="CLIP Extension Model(s):", multiselect=True)
                clip_ext_mode = gr.Radio(choices=["best", "fast", "classic", "negative"], value='best', label="CLIP Extension Mode")
                unload_clip_models_afterwords = gr.Checkbox(label="Unload CLIP Interrogator After Use", value=True)
                unload_clip_models_button = gr.Button(value="Unload All CLIP Interrogators")
                
            # WD EXT Options
            wd_ext_accordion = gr.Accordion("WD EXT Options:", visible=False)
            with wd_ext_accordion:
                wd_ext_model = gr.Dropdown(choices=[], value='wd-v1-4-moat-tagger.v2', label="WD Extension Model(s):", multiselect=True)
                wd_threshold = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Threshold")
                wd_underscore_fix = gr.Checkbox(label="Remove Underscores from Tags", value=True)
                unload_wd_models_afterwords = gr.Checkbox(label="Unload Tagger After Use", value=True)
                unload_wd_models_button = gr.Button(value="Unload All Tagger Models")
                    
            filtering_tools = gr.Accordion("Filtering tools:")
            with filtering_tools:
                use_positive_filter = gr.Checkbox(label="Filter Duplicate Positive Prompt Content from Interrogation")
                use_negative_filter = gr.Checkbox(label="Filter Duplicate Negative Prompt Content from Interrogation")
                use_custom_filter = gr.Checkbox(label="Filter Custom Prompt Content from Interrogation")
                custom_filter = gr.Textbox(value=self.load_custom_filter(),
                    label="Custom Filter Prompt",
                    placeholder="Prompt content separated by commas. Warning ignores attention syntax, parentheses '()' and colon suffix ':XX.XX' are discarded.",
                    show_copy_button=True                        
                )
                # Button to remove duplicates and strip strange spacing
                clean_custom_filter_button = gr.Button(value="Optimize Custom Filter")
                # Button to load/save custom filter from file
                with gr.Row():
                    load_custom_filter_button = gr.Button(value="Load Custom Filter")
                    save_confirmation_button = gr.Button(value="Save Custom Filter")
                save_confirmation_row = gr.Accordion("Are You Sure You Want to Save?", visible=False)
                with save_confirmation_row:
                    with gr.Row():
                        cancel_save_button = gr.Button(value="Cancel")
                        save_custom_filter_button = gr.Button(value="Save", variant="stop")
                
            experimental_tools = gr.Accordion("Experamental tools:", open=False)
            with experimental_tools:
                debug_mode = gr.Checkbox(label="Enable Debug Mode", info="[Debug Mode]: DEBUG statements will be printed to console log.")
                reverse_mode = gr.Checkbox(label="Enable Reverse Mode", info="[Reverse Mode]: Interrogation will be added to the negative prompt.")
                no_puncuation_mode = gr.Checkbox(label="Enable No Puncuation Mode", info="[No Puncuation Mode]: Interrogation will be filtered of all puncuations (except for a variety of emoji art).")
                exaggeration_mode = gr.Checkbox(label="Enable Exaggeration Mode", info="[Exaggeration Mode]: Interrogators will be permitted to add depulicate responses.")
                prompt_weight_mode = gr.Checkbox(label="Enable Interrogator Prompt Weight Mode", info="[Interrogator Prompt Weight]: Use attention syntax on interrogation.")
                prompt_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Interrogator Prompt Weight", visible=False) 
                prompt_output = gr.Checkbox(label="Enable Prompt Output", value=True, info="[Prompt Output]: Prompt statements will be printed to console log after every interrogation.")
                
            # Listeners
            model_selection.change(fn=self.update_clip_ext_visibility, inputs=[model_selection], outputs=[clip_ext_accordion, clip_ext_model])
            model_selection.change(fn=self.update_wd_ext_visibility, inputs=[model_selection], outputs=[wd_ext_accordion, wd_ext_model])
            unload_clip_models_button.click(self.unload_clip_models, inputs=None, outputs=None)
            unload_wd_models_button.click(self.unload_wd_models, inputs=None, outputs=None)
            prompt_weight_mode.change(fn=self.update_prompt_weight_visibility, inputs=[prompt_weight_mode], outputs=[prompt_weight])
            clean_custom_filter_button.click(self.clean_string, inputs=custom_filter, outputs=custom_filter)
            load_custom_filter_button.click(self.load_custom_filter, inputs=None, outputs=custom_filter)
            save_confirmation_button.click(self.update_save_confirmation_row_true, inputs=None, outputs=[save_confirmation_row])
            cancel_save_button.click(self.update_save_confirmation_row_false, inputs=None, outputs=[save_confirmation_row])
            save_custom_filter_button.click(self.save_custom_filter, inputs=custom_filter, outputs=[save_confirmation_row])
            refresh_models_button.click(fn=self.refresh_model_options, inputs=[], outputs=[model_selection])
                                    
        ui = [
            model_selection, debug_mode, in_front, prompt_weight_mode, prompt_weight, reverse_mode, exaggeration_mode, prompt_output, use_positive_filter, use_negative_filter, use_custom_filter, custom_filter, 
            clip_ext_model, clip_ext_mode, wd_ext_model, wd_threshold, wd_underscore_fix, unload_clip_models_afterwords, unload_wd_models_afterwords, no_puncuation_mode
            ]
        return ui

    def process_batch(
        self, p, model_selection, debug_mode, in_front, prompt_weight_mode, prompt_weight, reverse_mode, exaggeration_mode, prompt_output, use_positive_filter, use_negative_filter, use_custom_filter, custom_filter, 
        clip_ext_model, clip_ext_mode, wd_ext_model, wd_threshold, wd_underscore_fix, unload_clip_models_afterwords, unload_wd_models_afterwords, no_puncuation_mode, batch_number, prompts, seeds, subseeds):
            
        self.debug_print(debug_mode, f"process_batch called. batch_number={batch_number}, state.job_no={state.job_no}, state.job_count={state.job_count}, state.job_count={state.job}")
        if model_selection and not batch_number:
            # Calls reset_prompt_contamination to prep for multiple p.prompts
            if state.job_no <= 0:
                self.debug_print(debug_mode, f"Condition met for reset, calling reset_prompt_contamination")
                self.reset_prompt_contamination(debug_mode)
            #self.debug_print(debug_mode, f"prompt_contamination: {self.prompt_contamination}")
            # Experimental reverse mode cleaner
            if not reverse_mode:
                # Remove contamination from previous batch job from negative prompt
                p.prompt = p.prompt.replace(self.prompt_contamination, "")
            else:
                # Remove contamination from previous batch job from negative prompt
                p.negative_prompt = p.prompt.replace(self.prompt_contamination, "")
            
            # local variable preperations
            self.debug_print(debug_mode, f"Initial p.prompt: {p.prompt}")
            preliminary_interrogation = ""
            interrogation = ""
            
            # fix alpha channel
            init_image = p.init_images[0]
            p.init_images[0] = p.init_images[0].convert("RGB")
            
            # Interrogator interrogation loop
            for model in model_selection:
                # Check for skipped job
                if state.skipped:
                    print("Job skipped.")
                    state.skipped = False
                    continue
                    
                # Check for interruption
                if state.interrupted:
                    print("Job interrupted. Ending process.")
                    state.interrupted = False
                    break
                    
                # Should add the interrogators in the order determined by the model_selection list
                if model == "Deepbooru (Native)":
                    preliminary_interrogation = deepbooru.model.tag(p.init_images[0]) 
                    self.debug_print(debug_mode, f"[Deepbooru (Native)]: [Result]: {preliminary_interrogation}")
                    interrogation += f"{preliminary_interrogation}, "
                elif model == "CLIP (Native)":
                    preliminary_interrogation = shared.interrogator.interrogate(p.init_images[0]) 
                    self.debug_print(debug_mode, f"[CLIP (Native)]: [Result]: {preliminary_interrogation}")
                    interrogation += f"{preliminary_interrogation}, "
                elif model == "CLIP (EXT)":
                    if self.clip_ext is not None:
                        for clip_model in clip_ext_model:
                            # Clip-Ext resets state.job system during runtime...
                            job = state.job
                            job_no = state.job_no
                            job_count = state.job_count
                            # Check for skipped job
                            if state.skipped:
                                print("Job skipped.")
                                state.skipped = False
                                continue
                            # Check for interruption
                            if state.interrupted:
                                print("Job interrupted. Ending process.")
                                state.interrupted = False
                                break
                            preliminary_interrogation = self.clip_ext.image_to_prompt(p.init_images[0], clip_ext_mode, clip_model) 
                            if unload_clip_models_afterwords:
                                self.clip_ext.unload()
                            self.debug_print(debug_mode, f"[CLIP ({clip_model}:{clip_ext_mode})]: [Result]: {preliminary_interrogation}")
                            interrogation += f"{preliminary_interrogation}, "
                            # Redeclare variables for state.job system
                            state.job = job
                            state.job_no = job_no
                            state.job_count = job_count
                elif model == "WD (EXT)":
                    if self.wd_ext_utils is not None:
                        for wd_model in wd_ext_model:
                            # Check for skipped job
                            if state.skipped:
                                print("Job skipped.")
                                state.skipped = False
                                continue
                            # Check for interruption
                            if state.interrupted:
                                print("Job interrupted. Ending process.")
                                state.interrupted = False
                                break
                            rating, tags = self.wd_ext_utils.interrogators[wd_model].interrogate(p.init_images[0])
                            tags_list = [tag for tag, conf in tags.items() if conf > wd_threshold]
                            if wd_underscore_fix:
                                tags_spaced = [self.replace_underscores(tag) for tag in tags_list]
                                preliminary_interrogation = ", ".join(tags_spaced) 
                            else:
                                preliminary_interrogation = ", ".join(tags_list)
                            if unload_wd_models_afterwords:
                                self.wd_ext_utils.interrogators[wd_model].unload()
                            self.debug_print(debug_mode, f"[WD ({wd_model}:{wd_threshold})]: [Result]: {preliminary_interrogation}")
                            self.debug_print(debug_mode, f"[WD ({wd_model}:{wd_threshold})]: [Ratings]: {rating}")
                            interrogation += f"{preliminary_interrogation}, "
                            
            # Filter prevents overexaggeration of tags due to interrogation models having similar results 
            if not exaggeration_mode:
                interrogation = self.clean_string(interrogation)
            
            # Remove duplicate prompt content from interrogator prompt
            if use_positive_filter:
                interrogation = self.filter_words(interrogation, p.prompt)
            # Remove negative prompt content from interrogator prompt
            if use_negative_filter:
                interrogation = self.filter_words(interrogation, p.negative_prompt)
            # Remove custom prompt content from interrogator prompt
            if use_custom_filter:
                interrogation = self.filter_words(interrogation, custom_filter)

            # Experimental tool for removing puncuations, but commas and a variety of emojis
            if no_puncuation_mode:
                interrogation = self.remove_punctuation(interrogation)
            
            # This will weight the interrogation, and also ensure that trailing commas to the interrogation are correctly placed.
            if prompt_weight_mode:
                interrogation = f"({interrogation.rstrip(', ')}:{prompt_weight}), "
            else:
                interrogation = f"{interrogation.rstrip(', ')}, "
            
            # Experimental reverse mode prep
            if not reverse_mode:
                prompt = p.prompt
            else:
                prompt = p.negative_prompt
            
            # This will construct the prompt
            if prompt == "":
                prompt = interrogation
            elif in_front == "Append to prompt":
                interrogation = f", {interrogation}"
                prompt = f"{prompt}{interrogation}"
            else:
                prompt = f"{interrogation}{prompt}"
            
            # Experimental reverse mode assignment
            if not reverse_mode:
                """
                Note: p.prompt, p.all_prompts[0], and prompts[0]
                    To get A1111 to record the updated prompt, p.all_prompts needs to be updated.
                    But, in process_batch to update the stable diffusion prompt, prompts[0] needs to be updated.
                    prompts[0] are already parsed for extra network syntax,
                """
                p.prompt = prompt
                for i in range(len(p.all_prompts)):
                    p.all_prompts[i] = prompt
                # As far as I can tell, prompts is a list that is always 1, 
                # as it is just p.prompt without the extra network syntax
                # But since it is a list, I think it should be iterated through to future proof
                for i in range(len(prompts)):
                    prompts[i] = re.sub("[<].*[>]", "", prompt)
            else:
                p.negative_prompt = prompt
                for i in range(len(p.all_negative_prompts)):
                    p.all_negative_prompts[i] = prompt
                
            # Restore Alpha Channel
            p.init_images[0] = init_image
            
            # Prep for reset
            self.prompt_contamination = interrogation
            
            # Prompt Output default is True
            self.debug_print(prompt_output or debug_mode, f"[Prompt]: {prompt}")
            
            self.debug_print(debug_mode, f"End of {NAME} Process ({state.job_no+1}/{state.job_count})...")
        
#Startup Callbacks
script_callbacks.on_app_started(Script.load_clip_ext_module_wrapper)
script_callbacks.on_app_started(Script.load_wd_ext_module_wrapper)
