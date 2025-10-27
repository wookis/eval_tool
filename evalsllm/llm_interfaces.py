import os
import json
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod
from utils.logger import logger

from mmo_lr_utils.labeler.requester import get_dict_response_from_body  
import asyncio


def load_llm_model_info() -> Dict[str, Any]:
    """LLM 모델 정보를 JSON 파일에서 로드"""
    config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_model_info.json')
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"모델 정보 파일을 찾을 수 없습니다: {config_file}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 에러: {e}")
        return {}


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


class GenericLLM(LLMInterface):
    """JSON 설정을 기반으로 동적으로 생성되는 LLM 클래스"""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.model_config = model_config
        self.api_key = model_config.get('api_key')
        self.url = model_config.get('url')
        self.headers = model_config.get('headers', {})
        
        if not self.api_key:
            raise ValueError(f"{model_name}의 API 키가 설정되지 않았습니다.")
        if not self.url:
            raise ValueError(f"{model_name}의 URL이 설정되지 않았습니다.")

    def generate_response(self, prompt: str) -> Any:
        logger.debug(f"{self.model_name} 연동")
        try:
            # OpenAI 스타일 API 호출
            if 'openai' in self.url or 'azure' in self.url:
                return self._call_openai_style_api(prompt)
            else:
                # 기타 API 호출
                return self._call_generic_api(prompt)
        except Exception as e:
            logger.error(f"Error calling {self.model_name} API: {e}")
            return None

    def _call_openai_style_api(self, prompt: str) -> Any:
        """OpenAI 스타일 API 호출"""
        try:
            from openai import OpenAI
            
            # URL에서 base_url 추출
            if 'openai.azure.com' in self.url:
                # Azure OpenAI의 경우
                base_url = self.url.split('/openai/deployments/')[0]
            else:
                # 일반 OpenAI의 경우
                base_url = self.url.replace('/chat/completions', '').replace('?api-version=2025-01-01-preview', '')
            
            client = OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
            
            model_id = self.model_config.get('id', self.model_name)
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": ""}, 
                    {"role": "user", "content": prompt}
                ]
            )
            return response
        except ImportError:
            logger.error("OpenAI 라이브러리가 설치되지 않았습니다. 'pip install openai'를 실행해주세요.")
            return None

    def _call_generic_api(self, prompt: str) -> Any:
        """일반적인 HTTP API 호출"""
        print( """일반적인 HTTP API 호출""")
        try:
            body = {
                "messages": [
                    {"role": "system", "content": ""}, 
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = asyncio.run(get_dict_response_from_body(
                url=self.url,
                headers=self.headers,
                body=body
            ))
            return response
        except Exception as e:
            logger.error(f"Generic API 호출 실패: {e}")
            return None

    def get_model_name(self) -> str:
        return self.model_name

    def get_price_info(self) -> Dict[str, float]:
        """가격 정보 반환"""
        return self.model_config.get('price', {})


class LLMModelFactory:
    """LLM 모델을 동적으로 생성하는 팩토리 클래스"""
    
    def __init__(self):
        self.model_info = load_llm_model_info()
    
    def create_model(self, model_key: str) -> LLMInterface:
        """모델 키를 기반으로 LLM 인스턴스 생성"""
        if model_key not in self.model_info:
            raise ValueError(f"모델 '{model_key}'를 찾을 수 없습니다. 사용 가능한 모델: {list(self.model_info.keys())}")
        
        model_config = self.model_info[model_key]
        model_name = model_config.get('alias', model_key)
        
        return GenericLLM(model_name, model_config)
    
    def get_available_models(self) -> list:
        """사용 가능한 모델 목록 반환"""
        return list(self.model_info.keys())
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """특정 모델의 정보 반환"""
        if model_key not in self.model_info:
            raise ValueError(f"모델 '{model_key}'를 찾을 수 없습니다.")
        return self.model_info[model_key]


# 팩토리 인스턴스 생성
llm_factory = LLMModelFactory()


# 기존 호환성을 위한 클래스들 (deprecated)
class OpenAILLM(LLMInterface):
    """OpenAI GPT 모델과의 상호작용을 위한 클래스 (deprecated - GenericLLM 사용 권장)"""
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
            return model_response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

    def get_model_name(self) -> str:
        return self.model_name

# 기존 호환성을 위한 클래스들 (deprecated)
class KTModelLLM(LLMInterface):
    """KT Model 모델과의 상호작용을 위한 클래스 (deprecated - GenericLLM 사용 권장)"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
    
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
            return model_response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

    def get_model_name(self) -> str:
        return self.model_name

class AnthropicLLM(LLMInterface):
    """Anthropic Claude 모델과의 상호작용을 위한 클래스 (deprecated - GenericLLM 사용 권장)"""
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


# 사용 예시 함수
def create_llm_from_config(model_key: str) -> LLMInterface:
    """JSON 설정에서 모델을 생성하는 편의 함수"""
    return llm_factory.create_model(model_key)


def get_all_available_models() -> list:
    """사용 가능한 모든 모델 목록을 반환하는 편의 함수"""
    return llm_factory.get_available_models()


def get_model_price_info(model_key: str) -> Dict[str, float]:
    """특정 모델의 가격 정보를 반환하는 편의 함수"""
    model_info = llm_factory.get_model_info(model_key)
    return model_info.get('price', {})