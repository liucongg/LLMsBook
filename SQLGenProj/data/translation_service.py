from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class Translation():
    def __init__(self, model_path=None) -> None:
        
        if model_path is not None:
            model_path = model_path
        else:
            model_path = "./Helsinki-NLP/opus-mt-zh-en" # 需要下载中的翻译模型 本次采用HuggingFace 中的 Helsinki-NLP/opus-mt-zh-en 
        self.tokenizer_ez = AutoTokenizer.from_pretrained(model_path)
        self.model_ez = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
        print("Model init finished")

    def translate(self, text):
        # batch = self.tokenizer_ez.prepare_translation_batch(src_texts=[text])#3.0.2
        batch = self.tokenizer_ez.prepare_seq2seq_batch(src_texts=[text])#4.26.0
        # batch.to(device)
        batch["input_ids"] = torch.tensor(batch["input_ids"]).cuda()
        batch["attention_mask"] = torch.tensor(batch["attention_mask"]).cuda()

        translation = self.model_ez.generate(**batch)
        result = self.tokenizer_ez.batch_decode(translation, skip_special_tokens=True)
        return result
