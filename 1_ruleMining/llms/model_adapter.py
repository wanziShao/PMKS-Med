from .base_hf_causal_model import HfCausalModel
from .conv_prompt import *

class Llama(HfCausalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        conv = get_conv_template("llama-2")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()
    
    def prepare_for_inference(self):
        super().prepare_for_inference()
        self.maximun_token = 4096
    
    
class Mistral(HfCausalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        conv = get_conv_template("mistral")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()
    
    def prepare_for_inference(self):
        super().prepare_for_inference()
        self.maximun_token = 8192
    
class Qwen(HfCausalModel):
    DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        messages = [{"role": "user", "content": query}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    
class Vicuna(HfCausalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()
