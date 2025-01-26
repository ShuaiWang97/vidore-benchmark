from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
import os
from typing import Any, Dict, List, Optional

class QwenVLModel:
    def __init__(
        self,
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    ):
        # 设置使用的 GPU
        # os.environ["CUDA_VISIBLE_DEVICES"] = "4"

        # 配置 4-bit 量化
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # 加载模型
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="balanced_low_0",
            # quantization_config=quant_config,  # 启用量化配置
            # trust_remote_code=True,
        )

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def generate(self, messages, max_new_tokens=128):
        # 准备输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 确保所有输入都在同一个设备上
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # 生成输出
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]  # 返回第一个（也是唯一的）输出

    def rerank_with_vlm(self, query: str, images: List[Any]) -> List[int]:
        """
        Rerank images using Qwen-VL model and return indices of top 5 most relevant images
        
        Args:
            query: Query text
            images: List of images to rerank (typically top 10)
            
        Returns:
            List of indices for the top 5 most relevant images
        """
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"I will show you {len(images)} images. Your task is to identify the top 5 most relevant images to this query: '{query}'. Return a Python list containing the indices (0-{len(images)-1}) of the 5 most relevant images in order. For example: [2,5,1,8,3]"
                }
            ]
        }]
        
        # Add all images to the message
        for idx, image in enumerate(images):
            messages[0]["content"].append({
                "type": "image",
                "image": "assets/demo.jpeg"
            })
        
        try:
            # Generate response using the model
            output = self.model.generate(messages, max_new_tokens=32)
            
            # Find the list content between brackets
            start_idx = output.find('[')
            end_idx = output.rfind(']')
            if start_idx != -1 and end_idx != -1:
                list_str = output[start_idx + 1:end_idx]
                indices = [int(x.strip()) for x in list_str.split(',') if x.strip().isdigit()][:5]
                return indices if indices else list(range(min(5, len(images))))
                
            return list(range(min(5, len(images))))
        except Exception as e:
            print(f"Warning: Error in reranking: {str(e)}")
            # Fallback to first 5 indices if parsing fails
            return list(range(min(5, len(images))))    


# 使用示例：
if __name__ == "__main__":
    # 初始化模型
    model = QwenVLModel()

    # 准备输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "assets/demo.jpeg",
                },
                {
                    "type": "image",
                    "image": "assets/微信图片_20250125151912.png",
                },
                {
                    "type": "image",
                    "image": "assets/2.png",
                },
                {
                    "type": "text",
                    "text": "which picture has a car, tell me why and then output the picture index. Only output the picture index as python list format",
                },
            ],
        }
    ]

    # 获取输出
    output = model.generate(messages)
    print(output)
