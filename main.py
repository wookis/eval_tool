import sys
import os

from utils.logger import logger
from dotenv import load_dotenv
from utils.parser import load_dataset, eval_result_json
from utils.eda import create_eval_file, dataset_validation
from evalsllm.core import LLMJudge
from evalsllm.llm_interfaces import OpenAILLM
import pandas as pd
import numpy as np

from pathlib import Path

def main():
    logger.info("============== Starting eval-tool ==============")

    if len(sys.argv) != 3:
        print("Usage: python main.py <action> <file_path>")
        print("<action> : vaild / label / response / prompt / eval / metrics ")
        print("Example: python eval main.py dataset/sample.csv")
        sys.exit(1)

    action = sys.argv[1]
    if action == "eval":
        logger.info("평가를 진행합니다.")
    elif action == "label":
        logger.info("TLDC 라벨 생성을 진행합니다.")
        sys.exit(1)
    elif action == "response":
        logger.info("답변 생성을 진행합니다.")
        sys.exit(1)
    elif action == "prompt":
        logger.info("평가 프롬프트 생성을 진행합니다.")
        sys.exit(1)
    elif action == "metrics":
        logger.info("평가 메트릭 생성을 진행합니다.")
        sys.exit(1)
    else:
        logger.info("잘못된 작업입니다.")
        sys.exit(1)    

    file_path = sys.argv[2]

    try:
        #response 파일 로드
        logger.info(f"Input_file: {file_path}")
        df = load_dataset(file_path)
        # DataFrame이 비어있는지 확인
        if df.empty:
            logger.error("로드된 데이터가 비어있습니다.")
            return
        logger.info(f"Task, Response 결측치 제거 전 데이터 개수: {len(df)}")
        # 데이터 검증
        if input("데이터 검증 및 결측치 제거를 진행 하시겠습니까? (y/n): ") == "y":
            print("===========검증실행==========")
            df = dataset_validation("df_eval", df)
            logger.info(f"결측치 제거 후 데이터 개수: {len(df)}")
        else:
            print("검증안함")


        #eval 파일 생성
        df_eval = create_eval_file(df)

        file_ext = Path(file_path).suffix.lower()
        df_eval_file = file_path.replace(file_ext, "_df_eval.csv")        
        df_eval.to_csv(df_eval_file, index=False)
        df_eval_result_file = file_path.replace(file_ext, "_df_eval_result.csv")
  

        #eval judge 실행
        load_dotenv()
        judge = LLMJudge()
        gpt4_o = OpenAILLM(model_name="gpt-4o")
        #midm_mini_inst_2_3_1 = OpenAILLM(model_name="midm-mini-inst-2.3.1")
        judge.add_model(gpt4_o)
        # judge.add_model(midm-mini-inst-2.3.1)
        # judge.add_model(llama-3-1-74b)

        judge_model = gpt4_o
        #평가기준 추가시
        # judge.add_criteria(EvaluationCriteria(name="정확성", description="응답이 사실에 근거하고 정확한가?", weight=1.5))

        #6. 응답셋 생성 
        #logger.info(f"--- 데이터셋 기반 응답 생성 ---")
        #judge.generate_responses_on_dataset(dataset)
        
        #7. 데이터셋 기반 평가 실행
        #logger.info(f"--- 데이터셋 기반 평가 시작 ---")
        #judge.evaluate_responses_on_dataset(dataset, judge_model)

        """평가 결과 파일 확인 후 실행"""
        df_eval = judge.eval_result_check(df_eval, df_eval_result_file)

        """데이터셋 기반 평가 실행"""
        logger.info(f"============== 데이터셋 기반 평가 시작 ==============")
        df_eval_result = judge.run_evaluation_on_dataset(df_eval, judge_model, df_eval_result_file)
        #judge.eval_result_validation(df_eval, df_eval_result_file)
        logger.info(f"============== 데이터셋 기반 평가 완료 ==============")
       
    except Exception as e:
        logger.error(f"main 평가 중단 = {e}")


    df_eval_result = pd.read_csv(df_eval_result_file)
    df_eval_result = dataset_validation("df_eval_result", df_eval_result)
    #평가결과 검증
    if input("평가 완료: 평가 결과 검증을 진행 하시겠습니까? (y/n): ") == "y":
        logger.info(f"============== 검증 실행 ==============")
        #eval_data_validation(df_eval_result)
        dataset_validation("df_eval_result", df_eval_result)
    else:
        sys.exit(1)

    if input("평가 결과 시각화를 보시겠습니까? (y/n): ") == "y":
        # streamlit 실행
        os.system("streamlit run metrics/metrics.py") 
    else:
        sys.exit(1)




        
    # 1. LLM Judge 인스턴스 생성
    #judge = LLMJudge()

    # 2. 평가 기준 추가
    # # judge.add_criteria(EvaluationCriteria(name="정확성", description="응답이 사실에 근거하고 정확한가?", weight=1.5))
    # # judge.add_criteria(EvaluationCriteria(name="완결성", description="응답이 질문의 모든 부분을 다루는가?"))
    # # judge.add_criteria(EvaluationCriteria(name="스타일", description="지정된 스타일(예: 전문가, 친근함)을 잘 따르는가?", max_score=5.0))
    # judge.add_criteria(EvaluationCriteria(name="총점", description="각 평가 점수의 평균을 총점으로 0-1사이 소수점 2자리로 계산해주세요.", max_score=1.0))

    # # 3. 평가할 LLM 모델 및 데이터셋 추가
    # try:
    #     gpt4_o = OpenAILLM(model_name="gpt-4o")
    #     #judge.add_model(gpt4_o)
    #     #judge.add_model(midm-mini-inst-2.3.1)
    #     #judge.add_model(midm-pro-inst-2.3)
    #     #judge.add_model(llama-3-1-74b)

    #     # 4. Judge용 모델 설정 (예: GPT-4o를 심판으로 사용)
    #     # judge_model = gpt4_o

    #     # # 5. 평가 데이터셋 로드
    #     # #dataset = judge.load_dataset("dataset/3.eval_data/702-samples-eval_midm-pro-inst-2.3.csv")
    #     # dataset = judge.load_dataset("dataset/3.eval_data/702-samples-eval_gpt-4o.csv")
        
    #     # if dataset:
    #     #     #6. 응답셋 생성 
    #     #     #logger.info(f"--- 데이터셋 기반 응답 생성 ---")
    #     #     #judge.generate_responses_on_dataset(dataset)
            
    #     #     #7. 데이터셋 기반 평가 실행
    #     #     logger.info(f"--- 데이터셋 기반 평가 시작 ---")
    #     #     #judge.evaluate_responses_on_dataset(dataset, judge_model)
    #     #     judge.run_evaluation_on_dataset(dataset, judge_model)
            
    # except Exception as e:
    #     print("error =", e)


    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
    sys.exit(1)


if __name__ == "__main__":
    main()
