import gradio as gr
import re #For remove_attention regular expressions
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
        
        model_options = ["CLIP", "Deepbooru"]
        model_selection = gr.Dropdown(choices=model_options, label="Select Interrogation Model(s)", multiselect=True, value="Deepbooru")
        
        with gr.Accordion("Deepbooru tools:"):
            no_duplicates = gr.Checkbox(label="Filter Duplicate Prompt Content from Interrogation", value=False)
            use_negatives = gr.Checkbox(label="Filter Negative Prompt Content from Interrogation", value=False)
        return [in_front, prompt_weight, model_selection, use_weight, no_duplicates, use_negatives]


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
            raw_negative = p.negative_prompt
            interrogator = self.filter_words(interrogator, raw_negative)

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
