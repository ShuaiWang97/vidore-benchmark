import os
import base64
import requests
from typing import List, Any

class OpenAIImageRanker:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        """
        初始化 OpenAI 图像排序器
        
        参数:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            model: 使用的模型名称
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    @staticmethod
    def encode_image(image_path):
        """
        对图片文件进行 Base64 编码
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def encode_image_pil(image):
        """
        对 PIL Image 对象进行 Base64 编码
        """
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _prepare_messages(self, query, images, use_pil=False):
        """
        准备发送给API的消息内容
        
        参数:
            query: 查询文本
            images: 图片列表 (可以是文件路径列表或PIL Image对象列表)
            use_pil: 是否使用PIL Image对象
        """
        prompt = f"I will show you {len(images)} images. Your task is to identify the top 5 most relevant images to this query: '{query}'. Return a Python list containing the indices (0-{len(images)-1}) of the 5 most relevant images in order. For example: [2,5,1,8,3]. Important: only return the indices, no other text."
        
        user_content = [{"type": "text", "text": prompt}]
        base64_images = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encode_image_pil(img) if use_pil else self.encode_image(img)}",
                    "detail": "high",
                },
            }
            for img in images
        ]
        user_content.extend(base64_images)
        return [{"role": "user", "content": user_content}]

    def rank_images(self, query, image_dir):
        """
        对指定目录下的图片进行相关性排序
        
        参数:
            query: 查询文本
            image_dir: 图片目录路径
            
        返回:
            排序结果和完整的API响应
        """
        images = sorted(os.listdir(image_dir))
        image_paths = [os.path.join(image_dir, image) for image in images]
        
        payload = {
            "model": self.model,
            "messages": self._prepare_messages(query, image_paths, use_pil=False),
            "max_tokens": 1600,
            "temperature": 0,
            "seed": 2024,
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        result = response.json()
        return result["choices"][0]["message"]["content"], result

    def rerank_with_vlm(self, query: str, images: List[Any]) -> List[int]:
        """
        Rerank images using OpenAI GPT-4V model and return indices of top 5 most relevant images
        
        Args:
            query: Query text
            images: List of images to rerank (PIL Image objects)
            
        Returns:
            List of indices for the top 5 most relevant images
        """
        try:
            payload = {
                "model": self.model,
                "messages": self._prepare_messages(query, images, use_pil=True),
                "max_tokens": 1600,
                "temperature": 0,
                "seed": 2024,
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            result = response.json()
            output = result["choices"][0]["message"]["content"]
            
            # 解析输出获取索引
            start_idx = output.find('[')
            end_idx = output.rfind(']')
            if start_idx != -1 and end_idx != -1:
                list_str = output[start_idx + 1:end_idx]
                indices = [int(x.strip()) for x in list_str.split(',') if x.strip().isdigit()][:5]
                return indices if indices else list(range(min(5, len(images))))
            
            return list(range(min(5, len(images))))
            
        except Exception as e:
            print(f"Warning: Error in OpenAI reranking: {str(e)}")
            return list(range(min(5, len(images))))

# 使用示例
if __name__ == "__main__":
    query = "我想要买一辆车，你有什么推荐"
    image_dir = "assets"
    
    ranker = OpenAIImageRanker()
    result, full_response = ranker.rank_images(query, image_dir)
    print("Ranked indices:", result)
    # print("Full response:", full_response)
