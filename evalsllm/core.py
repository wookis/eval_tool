import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

import numpy as np
from datetime import datetime, timedelta
import csv
from tqdm import tqdm
import re
import pandas as pd
import yaml
from jinja2 import Template
import ast
from utils.logger import logger
from utils.parser import eval_result_json
from evalsllm.llm_interfaces import LLMInterface, OpenAILLM


PROMPT_TEMPLATES = {}

def load_prompt_templates(task : str):

    if task == 'T10'or task == 'T9':
        PROMPT_CONFIG_PATH = 'config/Knowledge_template.yaml'
    elif task == 'T3' or task == 'T4' or task == 'T8':
        PROMPT_CONFIG_PATH = 'config/Reason_template.yaml'
    elif task == 'T5' or task == 'T6' or task == 'T7':
        PROMPT_CONFIG_PATH = 'config/Creative_template.yaml'
    elif task == 'T2':
        PROMPT_CONFIG_PATH = 'config/Summary_template.yaml'
    elif task == 'T1' or task == 'T11' or task == 'T12':
        PROMPT_CONFIG_PATH = 'config/Reason_template.yaml'
    else:
        PROMPT_CONFIG_PATH = 'config/Reason_template.yaml'
        raise ValueError(f"Unknown task: {task}")

    try:
        with open(PROMPT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print("error load =", e)
        return {}

    

@dataclass
class EvaluationCriteria:
    """평가 기준을 정의하는 클래스"""
    name: str
    description: str
    max_score: float = 10.0
    weight: float = 1.0

@dataclass
class EvaluationResult:
    """평가 결과를 저장하는 클래스"""
    model_name: str
    criteria_name: str
    score: float
    feedback: str
    prompt: str
    response: str
    timestamp: str = datetime.now().isoformat()

class LLMJudge:
    """LLM의 성능을 평가하는 메인 클래스"""
    
    def __init__(self):
        self.evaluation_criteria: List[EvaluationCriteria] = []
        self.results: List[EvaluationResult] = []
        self.models: Dict[str, LLMInterface] = {}
        self.eval_start_time: Optional[datetime] = None
        self.eval_end_time: Optional[datetime] = None
        self.prompt_scores: Dict[str, Dict[str, float]] = {}
        self.dataset_filepath: Optional[str] = None
        self.target_id: Optional[str] = None
        self.start_index: Optional[int] = None
        self.end_index: Optional[int] = None
        self.actual_count: Optional[int] = None
    
    def add_criteria(self, criteria: EvaluationCriteria):
        """평가 기준 추가"""
        self.evaluation_criteria.append(criteria)
    
    def add_model(self, model: LLMInterface):
        """평가할 모델 추가"""
        self.models[model.get_model_name()] = model
    
    def evaluate_response(
        self, 
        no: str, 
        domain: str, 
        task: str, 
        level: str, 
        model_name: str, 
        token_usage: int,
        prompt: str,
        response: str,
        reference_data: Optional[str] = None,
        evaluation_prompt: Optional[str] = None,
        judge_model: Optional[LLMInterface] = None
    ) -> str:
        """특정 프롬프트에 대한 모델의 응답을 평가 (여러 기준을 한 번에 심판 모델에 묻는 구조)"""
        # if model_name not in self.models:
        #     raise ValueError(f"Model {model_name} not found")
        
        results = []
        output_prompt_file = "dataset/4.eval_result_data/250911-102500-prompt_df3.csv"
        # score_feedback 변수 초기화
        score_feedback = "Error: No response received"
        judge_response = "Error: No judge model provided"

        if judge_model:
            # task별 동적 평가 프롬프트 생성
            evaluation_prompt = self._create_multi_criteria_prompt(task,
                prompt, response, reference_data, self.evaluation_criteria
            )
            #logger.debug(f"eval prompt : {evaluation_prompt}")
            
            # Create DataFrame from prompt string for CSV saving
            prompt_df = pd.DataFrame({
                'no': [no],
                'task': [task],
                'domain': [domain],
                'level': [level],
                'prompt': [prompt],
                'response': [response],
                'reference_data': [reference_data or ''],
                'evaluation_prompt': [evaluation_prompt]
            })
            #기존파일에 추가
            prompt_df.to_csv(output_prompt_file, index=False, encoding='utf-8-sig', mode='a')
            #prompt_df.to_csv(output_prompt_file, index=False, encoding='utf-8-sig')
            #print(evaluation_prompt)
            judge_response = judge_model.generate_response(evaluation_prompt)
        
        return judge_response

    def _create_multi_criteria_prompt(self, task: str, prompt: str, response: str, reference_data: Optional[str], criteria_list: List[EvaluationCriteria]) -> str:
        """여러 평가 기준을 한 번에 묻는 프롬프트 생성"""
        #template_str = PROMPT_TEMPLATES['multi_criteria_prompt']
        # - Knowledge_template: 사실 정확성을 평가하는 프롬프트
        # - Reason_template: 단계별 설명의 품질을 평가하는 프롬프트
        # - Creative_template: 창작/서술형 콘텐츠의 품질을 평가하는 프롬프트
        # - Summary_template: 요약/요약형 평가 프롬프트
        # - Compare_template: 비교/분석 답변의 품질을 평가하는 프롬프트

        #PROMPT_CONFIG_PATH = 'config/Judge_template.yaml'
        PROMPT_TEMPLATES = load_prompt_templates(task)

        if task == 'T10'or task == 'T9':
            template_str = PROMPT_TEMPLATES['Knowledge_template']
        elif task == 'T3' or task == 'T4' or task == 'T8':
            template_str = PROMPT_TEMPLATES['Reason_template']
        elif task == 'T5' or task == 'T6' or task == 'T7':
            template_str = PROMPT_TEMPLATES['Creative_template']
        elif task == 'T2':
            template_str = PROMPT_TEMPLATES['Summary_template']
        elif task == 'T1' or task == 'T11' or task == 'T12':
            template_str = PROMPT_TEMPLATES['Reason_template']
        else:
            template_str = PROMPT_TEMPLATES['Reason_template']   
            raise ValueError(f"Unknown task: {task}")

        #PROMPT_TEMPLATES = load_prompt_templates(task)

        template = Template(template_str)
        return template.render(prompt=prompt, response=response, reference_answer=reference_data, criteria_list=criteria_list)

    def _parse_multi_criteria_judge_response(self, response: str, criteria_list: List[EvaluationCriteria]) -> List[tuple]:
        """심판 모델의 표/JSON 응답에서 각 기준별 점수와 피드백을 파싱 (JSON도 지원)"""
        results = []
        found_names = set()
        fail_reason = None
        # 1. JSON 배열 형태 시도
        try:
            json_array = json.loads(response)
            #print("json_array*********************", json_array)
            if isinstance(json_array, list):
                for obj in json_array:
                    name = obj.get('기준') or obj.get('criteria_name') or obj.get('name')
                    score = obj.get('점수') or obj.get('score')
                    feedback = obj.get('피드백') or obj.get('feedback')
                    try:
                        score = float(score)
                    except:
                        score = -1.0
                    results.append((name, score, feedback))
                    found_names.add(name)
        except Exception as e:
            fail_reason = f"JSON 배열 파싱 실패: {e}"
        # 2. 여러 개의 JSON 오브젝트가 줄바꿈으로 구분된 경우
        if not results:
            try:
                json_objs = re.findall(r'\{[^\{\}]+\}', response)
                for obj_str in json_objs:
                    try:
                        #print('\n', "***********************obj_str : ", json_objs, '\n')
                        obj = json.loads(obj_str.replace("'", '"'))
                        name = obj.get('기준') or obj.get('criteria_name') or obj.get('name')
                        score = obj.get('점수') or obj.get('score')
                        feedback = obj.get('피드백') or obj.get('feedback')
                        try:
                            #print('\n', "score count", score)
                            score = float(score)
                        except:
                            score = -1.0
                            #print('\n', "score error", score)
                        results.append((name, score, feedback))
                        found_names.add(name)
                    except Exception as e2:
                        fail_reason = f"JSON 오브젝트 파싱 실패: {e2, obj_str}"
                        # print('\n', name)
                        # print('\n', score)
                        # print('\n', feedback)
                        #print("****************e2 :", e2)
                        continue
            except Exception as e:
                fail_reason = f"JSON 오브젝트 전체 파싱 실패: {e}"
        # 3. 표 형태 파싱 (기존 방식)
        if not results:
            try:
                lines = [l.strip() for l in response.split('\n') if l.strip()]
                start = None
                for i, l in enumerate(lines):
                    if l.startswith('|') and '기준' in l and '점수' in l and '피드백' in l:
                        start = i + 1
                        break
                if start is not None:
                    for l in lines[start:]:
                        if not l.startswith('|'):
                            break
                        parts = [p.strip() for p in l.strip('|').split('|')]
                        if len(parts) < 3:
                            continue
                        name, score, feedback = parts[0], parts[1], parts[2]
                        try:
                            score = float(score)
                        except:
                            score = -1.0
                        results.append((name, score, feedback))
                        found_names.add(name)
            except Exception as e:
                fail_reason = f"표 파싱 실패: {e}"
        # 4. 모든 기준에 대해 결과가 없으면 -1점/실패 원인 메시지로 추가
        name_set = set(found_names)
        for c in criteria_list:
            if c.name not in name_set:
                msg = fail_reason or "심판 응답 파싱 실패 또는 기준 누락"
                results.append((c.name, -1.0, msg))
        # 5. 만약 결과가 아예 없으면 모든 기준 -1점 처리
        if not results:
            msg = fail_reason or "심판 응답 파싱 실패"
            results = [(c.name, -1.0, msg) for c in criteria_list]
        return results

    def _create_evaluation_prompt(self, 
                                prompt: str, 
                                response: str, 
                                reference_answer: Optional[str],
                                criteria: EvaluationCriteria) -> str:
        """평가를 위한 프롬프트 생성"""
        template_str = PROMPT_TEMPLATES['single_criteria_prompt']
        template = Template(template_str)
        return template.render(prompt=prompt, response=response, reference_answer=reference_answer, criteria=criteria)
    
    def _parse_judge_response(self, response: str) -> tuple[float, str]:
        """판단 모델의 응답을 파싱"""
        try:
            # 간단한 파싱 로직 - 실제 구현시 더 강건한 파싱 필요
            score_line = [l for l in response.split('\n') if l.startswith('점수:')][0]
            feedback_line = [l for l in response.split('\n') if l.startswith('피드백:')][0]
            
            score = float(score_line.split(':')[1].strip())
            feedback = feedback_line.split(':')[1].strip()
            
            return score, feedback
        except:
            return 0.0, "응답 파싱 실패"
    
    def _basic_evaluation(self, 
                         response: str, 
                         reference_answer: Optional[str],
                         criteria: EvaluationCriteria) -> tuple[float, str]:
        """기본적인 평가 로직"""
        if not reference_answer:
            return 0.0, "참조 답변 없이는 기본 평가를 수행할 수 없습니다."
        
        # 간단한 문자열 유사도 기반 평가 (실제 구현시 더 복잡한 평가 로직 필요)
        similarity = len(set(response.split()) & set(reference_answer.split())) / \
                    len(set(response.split()) | set(reference_answer.split()))
        
        score = similarity * criteria.max_score
        feedback = f"참조 답변과의 유사도: {similarity:.2f}"
        
        return score, feedback
    
    def get_results_summary(self) -> Dict[str, Any]:
        """평가 결과 요약"""
        if not self.results:
            return {"error": "평가 결과가 없습니다."}
        
        summary = {}
        for model_name in self.models.keys():
            model_results = [r for r in self.results if r.model_name == model_name]
            if not model_results:
                continue
                
            criteria_scores = {}
            for criteria in self.evaluation_criteria:
                criteria_results = [r for r in model_results if r.criteria_name == criteria.name]
                if criteria_results:
                    avg_score = np.mean([r.score for r in criteria_results])
                    criteria_scores[criteria.name] = {
                        "average_score": float(avg_score),
                        "weight": criteria.weight
                    }
            
            # 가중 평균 계산
            weights = [criteria_scores[c.name]["weight"] for c in self.evaluation_criteria 
                      if c.name in criteria_scores]
            scores = [criteria_scores[c.name]["average_score"] for c in self.evaluation_criteria 
                     if c.name in criteria_scores]
            
            if weights and scores:
                weighted_average = np.average(scores, weights=weights)
            else:
                weighted_average = 0.0
            
            summary[model_name] = {
                "criteria_scores": criteria_scores,
                "weighted_average": float(weighted_average)
            }
        
        # 평가 소요 시간 추가
        if self.eval_start_time and self.eval_end_time:
            duration = self.eval_end_time - self.eval_start_time
            duration_seconds = int(duration.total_seconds())
            duration_str = str(timedelta(seconds=duration_seconds))
        else:
            duration_str = None
        
        # prompt별 총점도 함께 반환
        prompt_score_summary = self.prompt_scores
        return {
            "summary": summary,
            "prompt_scores": prompt_score_summary,
            "evaluation_start_time": self.eval_start_time.isoformat() if self.eval_start_time else None,
            "evaluation_end_time": self.eval_end_time.isoformat() if self.eval_end_time else None,
            "duration": duration_str
        }
    
    def save_results(self, filepath: str):
        """평가 결과를 JSON 파일로 저장"""
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "results": [vars(r) for r in self.results],
            "summary": self.get_results_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

    def load_dataset(self, filepath: str) -> List[Dict[str, str]]:
        """CSV 파일에서 평가 데이터셋을 로드합니다."""
        self.dataset_filepath = filepath
        dataset = []
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset.append(row)
            print(f"'{filepath}'에서 {len(dataset)}개의 평가 항목을 로드했습니다.")
            return dataset
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다 - {filepath}")
            return []
        except Exception as e:
            print(f"오류: 데이터셋 로드 중 오류 발생 - {e}")
            return []

    def generate_responses_on_dataset(self, dataset: List[Dict[str, str]]):
        """데이터셋 전체에 대해 응답을 생성합니다."""
        self.eval_start_time = datetime.now()

        if not self.dataset_filepath:
            raise ValueError("데이터셋 파일 경로가 설정되지 않았습니다. load_dataset을 먼저 호출해주세요.")

        try:
            df = pd.read_csv(self.dataset_filepath, keep_default_na=False)
        except FileNotFoundError:
            df = pd.DataFrame(dataset)

        dataset_changed = False

        for index, row in tqdm(df.iterrows(), total=len(df), desc="응답 생성 및 저장"):
            prompt = row.get("prompt")
            if not prompt:
                continue

            item = row.to_dict()
            reference_answer = item.get("reference_answer")
            
            for model_name, model in self.models.items():
                response_col = f"{model_name}_response"
                if response_col not in df.columns:
                    df[response_col] = ""                    
                
                response = item.get(response_col)
                # print("\n ***********************************")
                # print("\n model : ", model_name, "response_col1 : ", response)
                # print("\n ***********************************")
                if not response:
                    print(f"응답 생성 중: {model_name} for '{prompt[:20]}...'")
                    response = model.generate_response(prompt)
                    df.loc[index, response_col] = response
                    item[response_col] = response
                    dataset_changed = True
                
        if dataset_changed:
            print(f"\n새로운 응답을 '{self.dataset_filepath}'에 저장합니다.")
            df.to_csv(self.dataset_filepath, index=False, encoding='utf-8-sig')

    def eval_result_check(self, df_eval: pd.DataFrame, df_eval_result_file: str):
        """평가 여부 확인"""
        #평가 결과 파일 존재 여부 확인
        target_id = None  # 함수 시작 부분에서 초기화
        start_index = 0
        
        if os.path.exists(df_eval_result_file):
            if input("평가 결과 파일이 존재합니다. 기존 평가를 이어서 하시겠습니까? (y/n): ") == "n":
                print(f"기존 파일삭제 후 새로 생성합니다.")
                os.remove(df_eval_result_file)
                # 새 파일 생성
                df_eval.to_csv(df_eval_result_file, index=False, encoding='utf-8-sig')
                print(f"새 파일 '{df_eval_result_file}'이 생성되었습니다.")
            else:
                df_eval_result = pd.read_csv(df_eval_result_file)
                print("df_eval_result", len(df_eval_result))
                print("df_eval", len(df_eval))

                # # df_eval과 df_eval_result를 id 기준으로 병합
                print("기존 평가 결과를 Merge 합니다...")
                df_eval = df_eval.merge(df_eval_result[['id', 'eval_prompt', 'eval_response', 'eval_result']], 
                                      on='id', how='left', suffixes=('', '_existing'))

                for i in range(len(df_eval_result)):
                    #print("df_eval_result.iloc[i].eval_result", i, df_eval_result.iloc[i].eval_response)
                    if df_eval_result.iloc[i].eval_response == "" or df_eval_result.iloc[i].eval_response is None or pd.isna(df_eval_result.iloc[i].eval_response):
                        print("여기부터다시", i, "target_id:", df_eval_result.iloc[i].id)
                        target_id = df_eval_result.iloc[i].id
                        df_eval = df_eval_result.copy()  # () 추가
                        break
                
                if not target_id:
                    print("모든 항목이 이미 평가되었습니다.")
                    return
        else:
            df_eval = df_eval.dropna(subset=["eval_result"])
            print(f"기존 파일이 없습니다. 새로 생성합니다.")
            # 새 파일 생성
            df_eval.to_csv(df_eval_result_file, index=False, encoding='utf-8-sig')
            print(f"새 파일 '{df_eval_result_file}'이 생성되었습니다.")
           

        print("target_id", target_id, "start_index", start_index)
        """해당 ID의 인덱스 찾기"""
        if target_id:  # target_id가 있는 경우에만 검색
            for idx, row in df_eval.iterrows():
                if row.get("id") == target_id:
                    start_index = idx
                    break

        if start_index is not None:
            print(f"ID '{target_id}'를 찾았습니다. 인덱스: {start_index}")
            logger.info(f"인덱스 {start_index}-{target_id}부터 다시 시작합니다.")
        else:
            start_index = 0

        # Set the instance variable for use in other methods
        self.start_index = start_index
              
        return df_eval
    

    def run_evaluation_on_dataset(self, 
                                  df_eval: List[Dict[str, str]], 
                                  judge_model: Optional[LLMInterface] = None,
                                  df_eval_result_file: Optional[str] = None):
        """데이터셋 전체에 대해 평가를 실행하고 응답을 캐싱합니다."""
        self.eval_start_time = datetime.now()

        start_index = getattr(self, 'start_index', 0) #3701
        df_eval_cnt = 3 #len(df_eval)-start_index
        
        end_index = min(start_index + df_eval_cnt, len(df_eval))
        actual_count = end_index - start_index
        
        logger.info(f"처리 범위: {start_index} ~ {end_index-1} (총 {actual_count}개)")
        
        if actual_count <= 0:
            print("처리할 데이터가 없습니다.")
            return
            
        for index, row in tqdm(df_eval.iloc[start_index:end_index].iterrows(), total=actual_count, desc="eval judge 실행"):

            # 이미 평가된 항목은 건너뛰기
            if (not pd.isna(row.get('eval_response')) and 
                row.get('eval_response') != "" and 
                row.get('eval_response') is not None):
                print(f"인덱스 {index}: 이미 평가된 항목 건너뛰기 (ID: {row.get('id')})")
                continue

            task_list = row.get("label_task")
            if isinstance(task_list, str):
                task_list = ast.literal_eval(task_list)

            if task_list is None or task_list == "":
                print("task_list is None or task_list == ", task_list)
                continue
            else:
                task = task_list[0]
            
            id = row.get("id")
            domain = row.get("domain")
            level = row.get("level")
            model_name = row.get("model_name")
            sp_max_tokens = row.get("sp_max_tokens")
            request = row.get("req")
            response = row.get("res")
            reference_data = row.get("ref")
            evaluation_criteria = ""

            #eval_prompt 생성          
            try:
                eval_prompt = self._create_multi_criteria_prompt(task,
                        request, response, reference_data, evaluation_criteria)
                #logger.info(f"eval_prompt 생성... : {index}{eval_prompt[:50]}")
            except Exception as e:
                logger.error(f"error eval_prompt 생성 = {e} {task}")
                continue

            #eval_response 생성
            try:
                eval_response = self.evaluate_response(
                    no = str(id) if id is not None else "",
                    domain = str(domain) if domain is not None else "",
                    task = str(task) if task is not None else "",
                    level = str(level) if level is not None else "",
                    model_name=model_name,
                    token_usage=sp_max_tokens,
                    prompt=request,
                    response=response,
                    reference_data=reference_data,
                    evaluation_prompt=eval_prompt,
                    judge_model=judge_model
                )
                #logger.info(f"eval_response 생성... : {index}{eval_response}")
                #logger.info(f"eval_response 생성... : {index}{request[:20]}")
            except Exception as e:
                logger.error(f"error eval_response 생성 = {e} {task}")
                continue

            df_eval.loc[index, "eval_prompt"] = eval_prompt
            df_eval.loc[index, "eval_response"] = str(eval_response)
         
            eval_result = eval_result_json(eval_response)
            df_eval.loc[index, "eval_result"] = eval_result

            if index!=0 and (index%5 == 0 or index == end_index-1):
                if df_eval_result_file:
                    logger.info(f"평가 저장 중: {index+1} - {id} for '{df_eval_result_file}...'")
                    df_eval.to_csv(df_eval_result_file, index=False, encoding='utf-8-sig')
                #del df_eval_result
        
        if df_eval_result_file:
            logger.info(f"평가 완료: {df_eval_result_file}, 총 {index+1}개의 평가 항목 처리")
        self.eval_end_time = datetime.now()
        logger.info(f"총 처리 시간: {self.eval_end_time - self.eval_start_time}")
        logger.debug(f"============== eval End ==============")
        
        return df_eval