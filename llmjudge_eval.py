import json
from llm_judge.core import LLMJudge, EvaluationCriteria
from llm_judge.llm_interfaces import OpenAILLM, KT_MAGMA_DEV_LLM
from dotenv import load_dotenv
from utils.logger import logger
from dataset.parser_result import parse_eval_feedback_to_results
from llm_judge.core import PROMPT_TEMPLATES 

load_dotenv()

def main():
    """LLM Judge 실행을 위한 메인 함수"""
    
    # 참고: 이 예제를 실행하려면 MODEL 환경 변수를 설정해야 합니다.
    
    # 1. LLM Judge 인스턴스 생성
    judge = LLMJudge()

    # 2. 평가 기준 추가
    # judge.add_criteria(EvaluationCriteria(name="정확성", description="응답이 사실에 근거하고 정확한가?", weight=1.5))
    # judge.add_criteria(EvaluationCriteria(name="완결성", description="응답이 질문의 모든 부분을 다루는가?"))
    # judge.add_criteria(EvaluationCriteria(name="스타일", description="지정된 스타일(예: 전문가, 친근함)을 잘 따르는가?", max_score=5.0))
    judge.add_criteria(EvaluationCriteria(name="총점", description="각 평가 점수의 평균을 총점으로 0-1사이 소수점 2자리로 계산해주세요.", max_score=1.0))

    # 3. 평가할 LLM 모델 및 데이터셋 추가
    try:
        gpt4_o = OpenAILLM(model_name="gpt-4o")
        judge.add_model(gpt4_o)
        #judge.add_model(midm-mini-inst-2.3.1)
        #judge.add_model(midm-pro-inst-2.3)
        #judge.add_model(llama-3-1-74b)

        # 4. Judge용 모델 설정 (예: GPT-4o를 심판으로 사용)
        judge_model = gpt4_o

        # 5. 평가 데이터셋 로드
        #dataset = judge.load_dataset("dataset/3.eval_data/702-samples-eval_midm-pro-inst-2.3.csv")
        dataset = judge.load_dataset("dataset/3.eval_data/702-samples-eval_gpt-4o.csv")
        
        if dataset:
            #6. 응답셋 생성 
            #logger.info(f"--- 데이터셋 기반 응답 생성 ---")
            #judge.generate_responses_on_dataset(dataset)
            
            #7. 데이터셋 기반 평가 실행
            logger.info(f"--- 데이터셋 기반 평가 시작 ---")
            #judge.evaluate_responses_on_dataset(dataset, judge_model)
            judge.run_evaluation_on_dataset(dataset, judge_model)
            
            #8. 결과 요약 및 저장
            logger.info("--- 평가 결과 파싱 시작 ---")
            csv_files = [
            "dataset/4.eval_result_data/702-samples-eval_feedback_midm-mini-inst-2.3.1.csv",
            "dataset/4.eval_result_data/702-samples-eval_feedback_midm-base-inst-2.3.2.csv",
            "dataset/4.eval_result_data/702-samples-eval_feedback_midm-pro-inst-2.3.csv",
            "dataset/4.eval_result_data/702-samples-eval_feedback_llama-3-1-74b.csv"
            ]
            output_file = "dataset/5.matrix_data/702-samples-eval_result_all.json"

            # 모든 CSV 파일을 합쳐서 하나의 JSON 파일로 저장
            results = parse_eval_feedback_to_results(csv_files, output_file)
            logger.info(f"파싱 완료: {len(results)}개의 결과 생성")

    except (ValueError, ImportError) as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    main() 