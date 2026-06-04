import os
from typing import List

from dotenv import load_dotenv
from zhipuai import ZhipuAI


def build_prompt(query: str, contexts: List[dict]) -> str:
    context_parts = []
    for idx, row in enumerate(contexts, start=1):
        context_parts.append(
            "\n".join(
                [
                    f"[证据 {idx}]",
                    f"chunk_id: {row['chunk_id']}",
                    f"来源: {row['title']} 第 {row['page']} 页",
                    f"内容: {row['text']}",
                ]
            )
        )

    return f"""你是一个严谨的科普事实核查助手。请只基于下面的证据回答问题。

要求：
1. 如果证据不足，明确说“证据不足”，不要编造。
2. 如果问题是在判断真假，请给出：结论、理由、引用证据。
3. 结论只能是：真实、虚假、部分真实、证据不足。
4. 每条关键判断必须引用证据编号，例如 [证据 1]。

用户问题：
{query}

检索证据：
{chr(10).join(context_parts)}
"""


class ZhipuGenerator:
    def __init__(self, model_name: str = "glm-4-flash"):
        load_dotenv()
        api_key = os.getenv("ZHIPUAI_API_KEY")
        self.model_name = model_name
        self.client = ZhipuAI(api_key=api_key) if api_key else None

    def generate(self, query: str, contexts: List[dict]) -> str:
        prompt = build_prompt(query, contexts)
        if not self.client:
            return "未配置 ZHIPUAI_API_KEY，已跳过 LLM 生成。\n\n" + prompt

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你只能基于用户提供的证据进行科普事实核查。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
