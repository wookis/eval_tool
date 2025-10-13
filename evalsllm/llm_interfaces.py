import os
from typing import Optional, Any
from abc import ABC, abstractmethod
from utils.logger import logger


class LLMInterface(ABC):
    """LLM과의 상호작용을 위한 추상 기본 클래스"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> Any:
        """프롬프트에 대한 응답을 생성"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """모델 이름 반환"""
        pass
from mmo_lr_utils.labeler.requester import get_dict_response_from_body  
import asyncio

# Note: To use these classes, you need to install the respective libraries:
# pip install openai anthropic

class OpenAILLM(LLMInterface):
    """OpenAI GPT 모델과의 상호작용을 위한 클래스"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI 라이브러리가 설치되지 않았습니다. 'pip install openai'를 실행해주세요.")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, prompt: str) -> Any:
        logger.debug(f"{self.model_name} 연동")
        try:
            model_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": ""}, 
                {"role": "user", "content": prompt}]
            )

            # # response 객체에서 content 추출
            # if model_response is not None:
            #     try:
            #         if hasattr(model_response, 'choices') and model_response.choices and hasattr(model_response.choices[0], 'message'):
            #             # OpenAI 응답 형식
            #             response = model_response.choices[0].message.content or ""
            #         elif hasattr(model_response, 'content') and model_response.content:
            #             # Anthropic 응답 형식
            #             if isinstance(model_response.content, list) and len(model_response.content) > 0:
            #                 response = model_response.content[0].text
            #             else:
            #                 response = str(model_response.content)
            #         else:
            #             response = str(model_response)
            #     except Exception as e:
            #         logger.error(f"Response parsing error: {e}")
            #         response = str(model_response)
            # else:
            #     response = "Error: Model returned None"

                
            #return model_response.choices[0].message.content or ""  
            return model_response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

    def get_model_name(self) -> str:
        return self.model_name

class KT_MAGMA_DEV_LLM(LLMInterface):
    """KT_MAGMA_DEV_LLM의 상호작용을 위한 클래스"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):

        self.api_key = api_key or os.getenv("KT_MAGMA_DEV_gemma_API_KEY")
        if not self.api_key:
            raise ValueError("KT_MAGMA_DEV_gemma_API_KEY가 설정되지 않았습니다.")
        
        self.model_name = model_name
        # TODO: 실제 KT_MAGMA_DEV API 클라이언트로 교체 필요
        # self.client = KT_MAGMA_DEV_Client(api_key=self.api_key)

        print(self.api_key)

    def generate_response(self, messages: list) -> Any:
        logger.debug(f"{self.model_name} 연동")
        try:
            url = "https://llm.prd.aiops-apim.kt.co.kr/midm-bitext-base-chat-OG085803-0/v1/chat/completions"
            headers = {
            "api-key": "fbe5953ba9e248fbaef2b79545e4b0de",
            "Content-Type": "application/json"
            }

            print("test get_dict_response_from_body")

            body = {
                "messages": messages,
            }
            response = asyncio.run(get_dict_response_from_body(
                **{
                    "url": url,
                    "headers": headers,
                    "body": body        
                }
                    )
            )
            
            print(response)
            
        except Exception as e:
            logger.error(f"Error calling KT_MAGMA_DEV API: {e}")
            return None

    def get_model_name(self) -> str:
        return self.model_name


class AnthropicLLM(LLMInterface):
    """Anthropic Claude 모델과의 상호작용을 위한 클래스"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic 라이브러리가 설치되지 않았습니다. 'pip install anthropic'를 실행해주세요.")
            
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
            
        self.model_name = model_name
        self.client = Anthropic(api_key=self.api_key)

    def generate_response(self, prompt: str) -> Any:
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return None

    def get_model_name(self) -> str:
        return self.model_name

