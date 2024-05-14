import gradio as gr
import re #For remove_attention regular expressions
from modules import scripts, deepbooru
from modules.processing import process_images
import modules.shared as shared
import os # Used for saving previous custom prompt

"""

Thanks to Mathias Russ.
Thanks to Smirking Kitsune.

"""


class Script(scripts.Script):

    def title(self):
        return "Img2img batch interrogator"

    def show(self, is_img2img):
        return is_img2img

    def b_clicked(o):
        return gr.Button.update(interactive=True)
    
    def ui(self, is_img2img):
        # Function to load custom filter from file
        def load_custom_filter(custom_filter):
            with open("extensions\sd-Img2img-batch-interrogator\custom_filter.txt", "r") as file:
                custom_filter = file.read()
                return custom_filter

        model_options = ["CLIP", "Deepbooru"]
        model_selection = gr.Dropdown(choices=model_options, label="Select Interrogation Model(s)", multiselect=True, value="Deepbooru")
        
        in_front = gr.Checkbox(label="Prompt in front", value=True)
        use_weight = gr.Checkbox(label="Use Interrogator Prompt Weight", value=True)
        prompt_weight = gr.Slider(
            0.0, 1.0, value=0.5, step=0.1, label="Interrogator Prompt Weight"
        )
        
        with gr.Accordion("Filtering tools:"):
            no_duplicates = gr.Checkbox(label="Filter Duplicate Prompt Content from Interrogation", value=False)
            use_negatives = gr.Checkbox(label="Filter Negative Prompt Content from Interrogation", value=False)
            use_custom_filter = gr.Checkbox(label="Filter Custom Prompt Content from Interrogation", value=False)
            custom_filter = gr.Textbox(
                label="Custom Filter Prompt", 
                placeholder="Prompt content seperated by commas. Warning ignores attention syntax, parentheses '()' and colon suffix ':XX.XX' are discarded.", 
                show_copy_button=True
            )
            # Button to load custom filter from file
            load_custom_filter_button = gr.Button(value="Load Last Custom Filter")
            
        # Listeners
        load_custom_filter_button.click(load_custom_filter, inputs=custom_filter, outputs=custom_filter)
        
        return [in_front, prompt_weight, model_selection, use_weight, no_duplicates, use_negatives, use_custom_filter, custom_filter]


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

    def run(self, p, in_front, prompt_weight, model_selection, use_weight, no_duplicates, use_negatives):

        raw_prompt = p.prompt
        interrogator = ""

        # fix alpha channel
        p.init_images[0] = p.init_images[0].convert("RGB")
        
        first = True # Two interrogator concatination correction boolean
        for model in model_selection:
            # This prevents two interrogators from being incorrectly concatinated
            if first == False:
                interrogator += ", "
            first = False
            # Should add the interrogators in the order determined by the model_selection list
            if model == "Deepbooru":
                interrogator += deepbooru.model.tag(p.init_images[0])
            elif model == "CLIP":
                interrogator += shared.interrogator.interrogate(p.init_images[0])
       
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
            with open("extensions\sd-Img2img-batch-interrogator\custom_filter.txt", "w") as file:
                file.write(custom_filter)

        if use_weight:
            if p.prompt == "":
                p.prompt = interrogator
            elif in_front:
                p.prompt = f"{p.prompt}, ({interrogator}:{prompt_weight})"
            else:
                p.prompt = f"({interrogator}:{prompt_weight}), {p.prompt}"
        else:
            if p.prompt == "":
                p.prompt = interrogator
            elif in_front:
                p.prompt = f"{p.prompt}, {interrogator}"
            else:
                p.prompt = f"{interrogator}, {p.prompt}"

        print(f"Prompt: {p.prompt}")

        processed = process_images(p)

        # Restore the UI elements we modified
        p.prompt = raw_prompt

        return processed
