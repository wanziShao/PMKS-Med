
from openai import OpenAI
import os
import time
from transformers import AutoTokenizer
from .start_fastchat_api import start_fastchat_api
import dotenv

dotenv.load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")

class LLMProxy(object):
    
    @staticmethod
    def regist_args(parser):
        parser.add_argument('--model_name', type=str, default='Qwen3-8B')
        parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B")
        parser.add_argument("--conv_template", type=str, default="qwen")
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--disable_auto_start", action="store_true")
        parser.add_argument('--retry', type=int, help="retry time", default=5)
        
    def __init__(self, args) -> None:
        self.args = args
        self.model_name = args.model_name
        if not args.disable_auto_start:
            start_fastchat_api(args.model_name, args.model_path, args.conv_template, args.host, args.port)
        self.retry = args.retry
        
    def prepare_for_inference(self):
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://{self.args.host}:{self.args.port}/v1",
        )
        self.client = client
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, token=HF_TOKEN,
        trust_remote_code=True,
        use_fast=False)
        self.maximun_token = self.tokenizer.model_max_length
    
    def token_len(self, text):
        """Returns the number of tokens used by a list of messages."""
        return len(self.tokenizer.tokenize(text))
    
    def generate_sentence(self, llm_input):
        query = [{"role": "user", "content": llm_input}]
        cur_retry = 0
        num_retry = self.retry
        # Chekc if the input is too long
        input_length = self.token_len(llm_input)
        if input_length > self.maximun_token:
            print(f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]
        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = query,
                    timeout=60,
                    temperature=0.0
                    )
                result = response.choices[0].message.content.strip() # type: ignore
                return result
            except Exception as e:
                print("Message: ", llm_input)
                print("Number of token: ", self.token_len(llm_input))
                print(e)
                time.sleep(30)
                cur_retry += 1
                continue
        return None
