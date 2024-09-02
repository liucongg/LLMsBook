import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Service:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).cuda()

    def predict(self, sql_info, query):
        """
         sql_info: 建表语句
         query: 用户问题
        """
        messages = [
            {'role': 'user', 'content': f"{sql_info}\n{query}"}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95,
                                      num_return_sequences=1,
                                      eos_token_id=self.tokenizer.eos_token_id)
        result = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return result


def parse_args():
    parser = argparse.ArgumentParser(description='train bpe tokenizer')
    parser.add_argument('--model_path', type=str, default="", help='保存的模型路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ss = Service(args.model_path)
