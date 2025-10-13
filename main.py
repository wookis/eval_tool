#from pydoc import resolve
import sys
import os
#from analysis import df_eval_result

from utils.logger import logger
from dotenv import load_dotenv
from tqdm import tqdm
from utils.parser import load_dataset, eval_result_json
from utils.eda import analyze_missing_values, analyze_data_distribution, dataset_validation
from utils.eda import create_eval_file
from evalsllm.core import LLMJudge, EvaluationCriteria
from evalsllm.llm_interfaces import OpenAILLM
from run_matrix import run_streamlit_matrix
from datetime import datetime
import pandas as pd
import ast
import numpy as np



#from evalsllm.core import PROMPT_TEMPLATES 

def main():
    logger.info("============== Starting eval-tool ==============")

    if len(sys.argv) != 3:
        print("Usage: python main.py <action> <file_path>")
        print("<action> : label / response / prompt / eval / metrics ")
        print("Example: python eval main.py dataset/sample.csv")
        sys.exit(1)

    # if sys.argv[1] == "eval" and input("평가를 진행할까요? (y/n): ") == "y":
    #     print("평가를 진행합니다.")
    # elif sys.argv[1] == "label" and input("답변을 생성할까요? (y/n): ") == "y":
    #     print("답변 생성을 진행합니다.")
    #     sys.exit(1)
    # elif sys.argv[1] == "prompt" and input("평가 프롬프트를 생성할까요? (y/n): ") == "y":
    #     print("평가 프롬프트 생성을 진행합니다.")
    # elif sys.argv[1] == "metrics" and input("평가 메트릭을 생성할까요? (y/n): ") == "y":
    #     print("평가 메트릭 생성을 진행합니다.")
    #     sys.exit(1)
    # else:
    #     print("잘못된 작업입니다.")
    #     sys.exit(1)

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
        #print(df_eval.head(10))
        df_eval_file = file_path.replace(".parquet", "_df_eval.csv")
        df_eval_result_file = file_path.replace(".parquet", "_df_eval_result.csv")
        #print(df_eval_file)
        df_eval.to_csv(df_eval_file, index=False)
  
        # print(df_eval.columns)
        # print(df['source_id'].value_counts())

        #eval judge 실행
        load_dotenv()
        judge = LLMJudge()
        gpt4_o = OpenAILLM(model_name="gpt-4o")
        judge.add_model(gpt4_o)
        judge_model = gpt4_o
        #평가기준 추가시
        # judge.add_criteria(EvaluationCriteria(name="정확성", description="응답이 사실에 근거하고 정확한가?", weight=1.5))

        #6. 응답셋 생성 
        #logger.info(f"--- 데이터셋 기반 응답 생성 ---")
        #judge.generate_responses_on_dataset(dataset)
        
        #7. 데이터셋 기반 평가 실행
        #logger.info(f"--- 데이터셋 기반 평가 시작 ---")
        #judge.evaluate_responses_on_dataset(dataset, judge_model)
        target_id = ""
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
                #df_eval = pd.concat([df_eval, df_eval_result], ignore_index=True)
                #df_eval = df_eval_result.copy()

                # 기존 평가 결과를 df_eval에 병합
                print("기존 평가 결과를 df_eval에 병합합니다...")
                
                # # df_eval과 df_eval_result를 id 기준으로 병합
                df_eval = df_eval.merge(df_eval_result[['id', 'eval_prompt', 'eval_response', 'eval_result']], 
                                      on='id', how='left', suffixes=('', '_existing'))

                # list_column_tolist = ["label_task", "label_level", "label_domain"]
                # for column in list_column_tolist:
                #     df_eval[column] = df_eval[column].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                
                # # 기존 평가 결과가 있는 컬럼들을 원본 컬럼으로 업데이트
                # df_eval['eval_prompt'] = df_eval['eval_prompt_existing'].fillna(df_eval['eval_prompt'])
                # df_eval['eval_response'] = df_eval['eval_response_existing'].fillna(df_eval['eval_response'])
                # df_eval['eval_result'] = df_eval['eval_result_existing'].fillna(df_eval['eval_result'])
                
                # # 임시 컬럼 제거
                # df_eval = df_eval.drop(columns=['eval_prompt_existing', 'eval_response_existing', 'eval_result_existing'])
                
                


                for i in range(len(df_eval_result)):
                    #print("existing_df.iloc[i].eval_result", i, existing_df.iloc[i].eval_response)
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
        
        # 해당 ID의 인덱스 찾기
        start_index = None
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

        judge.eval_start_time = datetime.now()

        df_eval_cnt = 3 #len(df_eval)-start_index #len(df_eval)
        #for index, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="eval judge 실행"):
        #for index, row in tqdm(df_eval[:df_eval_cnt].iterrows(), total=df_eval_cnt, desc="eval judge 실행"):
        #평가 실행 (특정 인덱스부터)
        #start_index = 3701
        
        # 안전한 범위 계산
        max_index = len(df_eval) - 1
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
                eval_prompt = judge._create_multi_criteria_prompt(task,
                        request, response, reference_data, evaluation_criteria)
                #logger.info(f"eval_prompt 생성... : {index}{eval_prompt[:50]}")
            except Exception as e:
                logger.error(f"error eval_prompt 생성 = {e} {task}")
                continue

            #eval_response 생성
            try:
                eval_response = judge.evaluate_response(
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

            # df_eval_result = pd.DataFrame({
            #     'id': id, 
            #     'source_id': row.get("source_id"),
            #     'label_task': task, 
            #     'label_level': row.get("label_level"), 
            #     'label_domain': row.get("label_domain"),
            #     'messages': row.get("messages"),
            #     'content': row.get("content"),
            #     'response': response,
            #     'answerer_llm_alias': row.get("answerer_llm_alias"),
            #     'status_code': row.get("status_code"),
            #     'ref': row.get("ref"),
            #     'req': row.get("req"),
            #     'res': row.get("res"),
            #     'eval_prompt': eval_prompt,
            #     'eval_response': str(eval_response),
            #     'eval_result': eval_result
            # })
            #logger.info(f"평가 결과: {eval_result}")
            if index !=0 and (index%5 == 0 or index == df_eval_cnt-1):
                logger.info(f"평가 저장 중: {index+1} - {id} for '{df_eval_result_file}...'")
                df_eval.to_csv(df_eval_result_file, index=False, encoding='utf-8-sig')
                #del df_eval_result
        
        logger.info(f"평가 완료: {df_eval_result_file}, 총 {index+1}개의 평가 항목 처리")
        judge.eval_end_time = datetime.now()
        logger.info(f"총 처리 시간: {judge.eval_end_time - judge.eval_start_time}")
        logger.info(f"============== eval End ==============")
        
       
    except Exception as e:
        print("error main =", e)
        logger.error(f"평가 중단 ")

    #평가결과 검증
    if input("평가 완료: 평가 결과 검증을 진행 하시겠습니까? (y/n): ") == "y":
        print("검증실행")
        #eval_data_validation(df_eval_result)
        dataset_validation("df_eval_result", df_eval)
    else:
        sys.exit(1)

    if input("평가 결과 시각화를 보시겠습니까? (y/n): ") == "y":
        # streamlit 실행
        os.system("streamlit run matrix/matrix.py") 
    else:
        sys.exit(1)

   

    #df_eval_result.csv 파일 로드
    # df_eval_result = pd.read_csv(eval_result_save_path)
    # print(df_eval_result.head(10))



        
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
