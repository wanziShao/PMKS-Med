from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .base_language_model import BaseLanguageModel
import os
import dotenv
dotenv.load_dotenv()

HF_TOKEN=os.getenv("HF_TOKEN")

class HfCausalModel(BaseLanguageModel):
    DEFAULT_MODEL_PATH = None
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--model_path', type=str, default=cls.DEFAULT_MODEL_PATH, help="HUGGING FACE MODEL or model path")
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=1024)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
        parser.add_argument('--quant', choices=["none", "4bit", "8bit"], default='none')
        parser.add_argument('--flash_atten_2', action='store_true', help="enable flash attention 2")
        
    def __init__(self, args):
        self.args = args
    
    def token_len(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self):
        if not self.args.model_path:
            raise ValueError("--model_path is required for Hugging Face models")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, token=HF_TOKEN,
        trust_remote_code=True, 
        use_fast=False)
        model_kwargs = {
            "device_map": "auto",
            "token": HF_TOKEN,
            "torch_dtype": self.DTYPE.get(self.args.dtype, None),
            "trust_remote_code": True,
        }
        if self.args.quant == "8bit":
            model_kwargs["load_in_8bit"] = True
        elif self.args.quant == "4bit":
            model_kwargs["load_in_4bit"] = True
        if self.args.flash_atten_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(self.args.model_path, **model_kwargs)
        self.generator = pipeline("text-generation", model=model, tokenizer=self.tokenizer)
        self.maximun_token = self.tokenizer.model_max_length
    
    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        model_prompt = self.prepare_model_prompt(llm_input)
        outputs = self.generator(
            model_prompt,
            return_full_text=False,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=False,
            handle_long_generation="hole",
        )
        return outputs[0]['generated_text'] # type: ignore
