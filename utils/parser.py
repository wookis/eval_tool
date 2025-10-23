import pandas as pd
import numpy as np

from utils.logger import logger
from ast import literal_eval
from pathlib import Path
import re
import json
import os


def load_llm_model_info():
    """LLM 모델 정보를 JSON 파일에서 로드"""
    config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_model_info.json')
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_file} 모델 정보를 로드할 수 없습니다.")
    except json.JSONDecodeError as e:
        print(f"Warning: JSON 파싱 에러: {e}")

def load_dataset(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from {file_path}")
    #dataset = pd.read_parquet(file_path)
    try:
        #확장자 확인
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            dataset = pd.read_csv(file_path, converters={
            'label_task': literal_eval,
            'label_level': literal_eval,
            'label_domain': literal_eval
        }).rename(columns={
            # 'label_task': 'task', 
            # 'label_level': 'level',
            # 'label_domain': 'domain',
            'model_name': 'answerer_llm_alias'
            })

            #dataset['label_task'] = dataset['label_task'].apply(lambda x: literal_eval(x))

            # dataset = dataset.dropna(subset=['label_task'])
            # dataset['label_task'] = dataset['label_task'].apply(lambda x: literal_eval(x))
            # dataset = dataset.dropna(subset=['label_level'])
            # dataset['label_level'] = dataset['label_level'].apply(lambda x: literal_eval(x))
            # dataset = dataset.dropna(subset=['label_domain'])
            # dataset['label_domain'] = dataset['label_domain'].apply(lambda x: literal_eval(x))
            # print(type(dataset['label_task'].iloc[0]))

            #print(dataset.columns)
            #dataset = dataset.rename(columns={"model_name":"answerer_llm_alias", "task":"label_task", "level":"label_level", "domain":"label_domain"})
            #dataset = pd.read_csv(file_path, converters={"messages":literal_eval})
        elif file_ext == '.pkl':            
            dataset = pd.read_pickle(file_path)
            dataset = dataset.dropna(subset=['label_task'])

            if type(dataset['label_task']) == str:
                dataset['label_task'] = dataset['label_task'].apply(lambda x: literal_eval(x))
                dataset = dataset.dropna(subset=['label_level'])
                dataset['label_level'] = dataset['label_level'].apply(lambda x: literal_eval(x))
                dataset = dataset.dropna(subset=['label_domain'])
                dataset['label_domain'] = dataset['label_domain'].apply(lambda x: literal_eval(x))

            dataset = dataset.rename(columns={"model_name":"answerer_llm_alias", "task":"label_task", "level":"label_level", "domain":"label_domain"})
            #

        elif file_ext == '.parquet':            
            dataset = pd.read_parquet(file_path)
        elif file_ext == '.json':            
            with open(file_path, encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_ext}")

        list_column_tolist = ["label_task", "label_level", "label_domain"]

        for column in list_column_tolist:
            dataset[column] = dataset[column].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        logger.info(f"'{file_path}'에서 {len(dataset)}개의 평가 항목을 로드했습니다.")
        return dataset

    except FileNotFoundError:
        logger.error(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"오류: 데이터셋 로드 중 오류 발생 - {e}")
        return pd.DataFrame()

def parse_json_block_safely(md_string: str) -> dict:
    # 1. 코드 블록 제거
    cleaned = re.sub(r'^```json\s*|\s*```$', '', md_string.strip(), flags=re.IGNORECASE)

    # 2. 첫 번째 시도: 그냥 파싱
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass  # 아래에서 재시도

    # 3. 두 번째 시도: \Delta 같은 수식용 백슬래시 escape
    def escape_problematic_backslashes(s):
        return re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', s)

    escaped = escape_problematic_backslashes(cleaned)

    # 4. 다시 파싱
    return json.loads(escaped)

def eval_result_json(eval_response):

    #print("eval_response_json", eval_response)
    # response 객체에서 content 추출
    if eval_response is not None:
        try:
            if hasattr(eval_response, 'choices') and eval_response.choices and hasattr(eval_response.choices[0], 'message'):
                # OpenAI 응답 형식
                response = eval_response.choices[0].message.content or ""
            elif hasattr(eval_response, 'content') and eval_response.content:
                # Anthropic 응답 형식
                if isinstance(eval_response.content, list) and len(eval_response.content) > 0:
                    response = eval_response.content[0].text
                else:
                    response = str(eval_response.content)
            else:
                response = str(eval_response)
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            response = str(eval_response)
    else:
        response = "Error: Model returned None"
    #print("eval_result_json", response)

    return response


def dataset_clean(df: pd.DataFrame):


    #df 결측값 확인
    #print(df.isnull().sum())

    # 결측치 데이터 제거
    print("결측치 데이터 제거 전", df.shape)

    #label_task 결측값 확인 및 제거
    #print("label_task 없는 데이터 제거 전", df.shape)
    print(df["answerer_llm_alias"].value_counts())
    print("label_task 없는 데이터 수", df[df['label_task'].isna()].shape)
    df = df[df['label_task'].notna()]
    #print("label_task 없는 데이터 제거 후", df.shape)
    #print(df["answerer_llm_alias"].value_counts())

    #answerer_llm_alias 결측값 확인 및 제거
    #print("answerer_llm_alias 없는 데이터 제거 전", df.shape)
    print(df["answerer_llm_alias"].value_counts())
    print("answerer_llm_alias 없는 데이터 수", df[df['answerer_llm_alias'].isna()].shape)
    df = df[df['answerer_llm_alias'].notna()]
    #print("answerer_llm_alias 없는 데이터 제거 후", df.shape)
    #print(df["answerer_llm_alias"].value_counts())

    #eval_result 결측값 확인 및 제거
    #print("eval_result 없는 데이터 제거 전", df.shape)
    print(df["answerer_llm_alias"].value_counts())
    print("eval_result 없는 데이터 수", df[df['eval_result'].isna()].shape)
    df = df[df['eval_result'].notna()]
    #print("eval_result 없는 데이터 제거 후", df.shape)
    #print(df["answerer_llm_alias"].value_counts())

    #df_filtered answerer_llm_alias 결측값 제거
    #print("LLM 정보 없는 데이터 제거 전", df_filtered.shape)
    print("LLM 정보 없는 데이터 수", df[df['answerer_llm_alias'].isna()].shape)
    df = df[df['answerer_llm_alias'].notna()]
    #print("LLM 정보 없는 데이터 제거 후", df_filtered.shape)
    print("answerer_llm_alias unique", df['answerer_llm_alias'].unique())

    print("결측치 데이터 제거 후", df.shape)

   
    #print(df.isnull().sum())
    
    # LLM 모델 정보 로드
    dict_llm_name2info = load_llm_model_info()
    dict_llm_alias2price = {
        llm_info['alias']:llm_info['price'] for llm_info in dict_llm_name2info.values()
    }

    dict_llm_alias2llm_id = {
        llm_info['alias']:llm_info['id'] for llm_info in dict_llm_name2info.values()
    }

    print(df[df['eval_result'].isna()]['lr_answer_bronze_id'].value_counts())
 
    

    list_error_index = []
    i=0
    for index, row in df.iterrows():
        try:
            dict_evaluation = parse_json_block_safely(row['eval_result'])
        except Exception as e:
            list_error_index.append(index)
            #df = df.drop(index=index)
            i += 1
            # print('----error-----')
            # print(e)
            # print(index)
            # print(row['eval_result'])
            
    print("Judge 결과 파싱 오류 개수 : ", i)
  
    #print(f"{df.shape=}")
    df_filtered = df.drop(index=list_error_index)
    
    #model_name 컬럼이름 변경
    df_filtered.rename(columns={'model_name': 'answerer_llm_alias'}, inplace=True)
    #print("df_filtered", df_filtered.shape)

    # lr_answer_bronze_id col이 있으면 실행
    if 'lr_answer_bronze_id' in df_filtered.columns:
        print('작업 전2')
        print(df_filtered['lr_answer_bronze_id'].value_counts().value_counts())
        df_filtered = df_filtered.groupby('lr_answer_bronze_id').filter(lambda x: len(x) == 5)
        print('작업 후2')
        print(df_filtered['lr_answer_bronze_id'].value_counts().value_counts())
        df_filtered.reset_index(drop=True, inplace=True)
        #df_filtered = df_filtered.groupby('lr_answer_bronze_id')

        df_filtered = df.copy()
        print(df_filtered.shape)
        print(len(df_filtered['lr_answer_bronze_id'].unique()))

        for index_to_remove in list_error_index:
            lr_answer_bronze_id_to_remove = df.loc[index_to_remove]['lr_answer_bronze_id']
            df_filtered = df_filtered[df_filtered['lr_answer_bronze_id'] != lr_answer_bronze_id_to_remove]

        df_filtered.reset_index(drop=True, inplace=True)

        print("lr_answer_bronze_id 없는 데이터 제거 후", df_filtered.shape)
    else:
        df['answerer_llm_id'] = df['answerer_llm_alias'].map(lambda model_name: dict_llm_alias2llm_id[model_name])
        import uuid
        df["lr_answer_bronze_id"] = df.groupby("messages")["content"].transform(
            lambda x: str(uuid.uuid4())
        )
      

    #수정버전 평가결과 파싱
    def safe_parse_quality(x):
        try:
            if pd.isna(x) or x == '' or x is None:
                return None
            parsed = parse_json_block_safely(x)
            return parsed.get('총점', None)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    df_filtered['quality'] = df_filtered['eval_result'].map(lambda x: parse_json_block_safely(x)['총점'])
    #df_filtered['quality'] = df_filtered['eval_result'].map(safe_parse_quality)
    
    # quality가 None인 행들을 제거
    df_filtered = df_filtered.dropna(subset=['quality'])
    df_filtered.reset_index(drop=True, inplace=True)
    ############################################################
    df_filtered['quality'].describe()
    
    json.loads(df_filtered.iloc[0]['response'])['usage']['completion_tokens']
    json.loads(df_filtered.iloc[0]['response'])['usage']['completion_tokens']

    print("사용 가능한 모델들:", [key for key in dict_llm_name2info])
    print("실제 데이터의 모델들:", df_filtered['answerer_llm_alias'].unique())

    
    
    # Quality 확인
    # for g in df_filtered["answerer_llm_alias"].unique():
    #     print(g)
    #     print(df_filtered[lambda row: row['answerer_llm_alias'] == g].describe())

    ##### Cost 계산
    list_cost = []

    for index, row in df_filtered.iterrows():
        try:
            answerer_llm_alias = row['answerer_llm_alias']
            #print("answerer_llm_alias", answerer_llm_alias)
            
            #모델명 매핑 (더 많은 케이스 추가)
            if answerer_llm_alias == 'gpt-4.1':
                answerer_llm_alias = 'gpt_4_1'
            elif answerer_llm_alias == 'midm-base-2.3.2':
                answerer_llm_alias = 'midm_base_2_3_2'
            elif answerer_llm_alias == 'gpt-4.1-mini':
                answerer_llm_alias = 'gpt_4_1_mini'
            elif answerer_llm_alias == 'midm-mini-2.3.1':
                answerer_llm_alias = 'midm_mini_2_3_1'
            elif answerer_llm_alias == 'midm-pro-inst':
                answerer_llm_alias = 'midm_pro'
            elif answerer_llm_alias == 'sota-k':
                answerer_llm_alias = 'sota_k'
            elif answerer_llm_alias == 'gpt-oss-120b':
                answerer_llm_alias = 'gpt_oss_120b'  # 기본값으로 매핑
            elif answerer_llm_alias == 'gpt-4o-mini':
                answerer_llm_alias = 'gpt_4_1_mini'  # 기본값으로 매핑
            elif answerer_llm_alias == 'llama-3.1-74b-fp16':
                answerer_llm_alias = 'llama_74b_fp16'  # 기본값으로 매핑
   
            # 모델 정보가 있는지 확인
            if answerer_llm_alias not in dict_llm_name2info:
                logger.error(f"Warning: {answerer_llm_alias} 모델 정보가 없습니다. 기본값 사용.")
                cost = 0.0
            else:
                dollar_per_input_token = dict_llm_name2info[answerer_llm_alias]['price']['dollar_per_input_token']
                dollar_per_output_token = dict_llm_name2info[answerer_llm_alias]['price']['dollar_per_output_token']
            
                # response가 JSON 문자열인지 확인
                if isinstance(row['response'], str):
                    response_data = json.loads(row['response'])
                    num_input_tokens = response_data['usage']['prompt_tokens']
                    num_output_tokens = response_data['usage']['completion_tokens']
                else:
                    # 이미 딕셔너리인 경우
                    num_input_tokens = row['response']['usage']['prompt_tokens']
                    num_output_tokens = row['response']['usage']['completion_tokens']

                cost = num_input_tokens * dollar_per_input_token + num_output_tokens * dollar_per_output_token
            
            list_cost.append(cost)
            
        except (KeyError, json.JSONDecodeError, TypeError) as e:
            print(f"Warning: 행 {index}에서 비용 계산 실패: {e}")
            print(f"  모델명: {row.get('answerer_llm_alias', 'Unknown')}")
            print(f"  Response 타입: {type(row.get('response', None))}")
            list_cost.append(0.0)

    df_filtered['cost'] = list_cost

    # for column in ['metadata', 'messages', 'answer', 'task', 'domain', 'level', 'judge_total_score', 'total_price']:
    #     if column in df.columns:
    #         print(f"column in {column}")
    #     else:
    #         print(f"column not in {column}")
            

    print("dataset_clean : ", df_filtered.shape)

    return df_filtered





import uuid
import json
from ast import literal_eval

def get_target_data(input_dir: str) -> pd.DataFrame:
    list_root_dir = input_dir
    taget_file = {}

    for file in list_root_dir.glob('702*.csv'):
        print(file)
        model = file.stem.split('eval_')[1].split('.csv')[0]
        taget_file[model] = pd.read_csv(file)
        print(model, " : ", len(taget_file[model]))
        print("--------------------------------")
        
    # list(sorted(dict_llm2df.keys()))


    # dict_llm2df['midm-mini-inst-2.3.1'].head(3)

def get_target_file_to_dict(input_file: str) -> dict:

    target_dict = {}
    df = pd.read_csv(input_file, converters={'messages': literal_eval})
    
    
    for index, row in df.iterrows():
        id = uuid.uuid4()

        target_dict[id] = {
            'task': row['task'],
            'level': row['level'],
            'domain': row['domain'],
            'messages': row['messages'],
        }

    return target_dict






# def parse_csv_file(file_path: str) -> List[Dict[str, Any]]:
#     """
#     CSV 파일을 파싱하여 messages의 content를 추출합니다.
    
#     Args:
#         file_path (str): 파싱할 CSV 파일 경로
        
#     Returns:
#         List[Dict[str, Any]]: 파싱된 데이터 리스트
#     """
#     try:
#         # CSV 파일 읽기
#         df = pd.read_csv(file_path, encoding='utf-8')
#         logger.info(f"CSV 파일 로드 완료: {file_path}, 컬럼: {list(df.columns)}")
        
#         parsed_data = []
        
#         # 컬럼명 확인 및 content 컬럼 찾기
#         content_columns = []
#         for col in df.columns:
#             if 'content' in col.lower() or 'message' in col.lower() or 'text' in col.lower():
#                 content_columns.append(col)
        
#         if not content_columns:
#             # content 관련 컬럼이 없으면 첫 번째 텍스트 컬럼 사용
#             for col in df.columns:
#                 if df[col].dtype == 'object':  # 문자열 타입 컬럼
#                     content_columns.append(col)
#                     break
        
#         if not content_columns:
#             logger.warning(f"CSV 파일에서 content 컬럼을 찾을 수 없습니다: {file_path}")
#             return []
        
#         logger.info(f"사용할 content 컬럼: {content_columns}")
        
#         # 각 행 처리
#         for index, row in df.iterrows():
#             for content_col in content_columns:
#                 if pd.notna(row[content_col]) and str(row[content_col]).strip():
#                     content_value = str(row[content_col]).strip()
                    
#                     # 메타데이터 추출
#                     metadata = {}
#                     for col in df.columns:
#                         if col != content_col and pd.notna(row[col]):
#                             metadata[col] = row[col]
                    
#                     parsed_data.append({
#                         'file_path': file_path,
#                         'row_index': index,
#                         'message_content': content_value,
#                         'content_column': content_col,
#                         'metadata': metadata
#                     })
        
#         logger.info(f"CSV 파일 {file_path}에서 {len(parsed_data)}개의 메시지 추출")
#         return parsed_data
        
#     except Exception as e:
#         logger.error(f"CSV 파일 {file_path} 파싱 중 오류 발생: {e}")
#         return []


# def parse_tldc_file(file_path: str) -> List[Dict[str, Any]]:
#     """
#     TLDC 데이터 파일을 파싱하여 messages의 content를 추출합니다.
#     파일 확장자에 따라 적절한 파서를 선택합니다.
    
#     Args:
#         file_path (str): 파싱할 파일 경로
        
#     Returns:
#         List[Dict[str, Any]]: 파싱된 데이터 리스트
#     """
#     file_ext = Path(file_path).suffix.lower()
    
#     if file_ext == '.csv':
#         return parse_csv_file(file_path)
#     elif file_ext == '.json':
#         return parse_json_file(file_path)
#     else:
#         logger.warning(f"지원하지 않는 파일 형식입니다: {file_ext}")
#         return []


# def parse_json_file(file_path: str) -> List[Dict[str, Any]]:
#     """
#     JSON 파일을 파싱하여 messages의 content를 추출합니다.
    
#     Args:
#         file_path (str): 파싱할 JSON 파일 경로
        
#     Returns:
#         List[Dict[str, Any]]: 파싱된 데이터 리스트
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         parsed_data = []
        
#         # 데이터 구조에 따라 messages content 추출
#         if isinstance(data, list):
#             # 리스트 형태의 데이터
#             for item in data:
#                 if isinstance(item, dict) and 'messages' in item:
#                     messages = item['messages']
#                     if isinstance(messages, list):
#                         for msg in messages:
#                             if isinstance(msg, dict) and 'content' in msg:
#                                 parsed_data.append({
#                                     'file_path': file_path,
#                                     'message_content': msg['content'],
#                                     'message_role': msg.get('role', 'unknown'),
#                                     'timestamp': msg.get('timestamp', ''),
#                                     'metadata': {k: v for k, v in msg.items() 
#                                                if k not in ['content', 'role', 'timestamp']}
#                                 })
#         elif isinstance(data, dict):
#             # 딕셔너리 형태의 데이터
#             if 'messages' in item:
#                 messages = data['messages']
#                 if isinstance(messages, list):
#                     for msg in messages:
#                         if isinstance(msg, dict) and 'content' in msg:
#                             parsed_data.append({
#                                 'file_path': file_path,
#                                 'message_content': msg['content'],
#                                 'message_role': msg.get('role', 'unknown'),
#                                 'timestamp': msg.get('timestamp', ''),
#                                 'metadata': {k: v for k, v in msg.items() 
#                                            if k not in ['content', 'role', 'timestamp']}
#                             })
        
#         logger.info(f"JSON 파일 {file_path}에서 {len(parsed_data)}개의 메시지 추출")
#         return parsed_data
        
#     except Exception as e:
#         logger.error(f"JSON 파일 {file_path} 파싱 중 오류 발생: {e}")
#         return []


# def save_response_to_file(response_data: Dict[str, Any], output_dir: str, 
#                          original_filename: str) -> str:
#     """
#     응답 데이터를 파일로 저장합니다.
    
#     Args:
#         response_data (Dict[str, Any]): 저장할 응답 데이터
#         output_dir (str): 출력 디렉토리 경로
#         original_filename (str): 원본 파일명
        
#     Returns:
#         str: 저장된 파일 경로
#     """
#     try:
#         # 출력 디렉토리 생성
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 파일명 생성 (원본 파일명에 _response 추가)
#         base_name = Path(original_filename).stem
#         output_filename = f"{base_name}_response.json"
#         output_path = os.path.join(output_dir, output_filename)
        
#         # 응답 데이터에 메타데이터 추가
#         response_data['metadata'] = {
#             'original_file': original_filename,
#             'processed_at': str(Path.cwd()),
#             'version': '1.0'
#         }
        
#         # JSON 파일로 저장
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(response_data, f, ensure_ascii=False, indent=2)
        
#         logger.info(f"응답이 저장되었습니다: {output_path}")
#         return output_path
        
#     except Exception as e:
#         logger.error(f"응답 저장 중 오류 발생: {e}")
#         return ""


#eval_response
          #print('\n', "***********************score_feedback : ", score_feedback, '\n')
            #print('\n', "***********************evaluation_prompt : ", evaluation_prompt, '\n')
            #logger.info(f"score_feedback : {score_feedback}")

        #     try:
        #         criteria_results = self._parse_multi_criteria_judge_response(score_feedback, self.evaluation_criteria)
        #     except Exception as e:
        #         # 파싱 실패 시 -1점 처리
        #         criteria_results = [
        #             (criteria.name, -1.0, f"파싱 실패: {e}") for criteria in self.evaluation_criteria
        #         ]
        #     for criteria_name, score, feedback in criteria_results:
        #         result = EvaluationResult(
        #             model_name=model_name,
        #             criteria_name=criteria_name,
        #             score=score,
        #             feedback=feedback,
        #             prompt=prompt,
        #             response=response
        #         )
        #         results.append(result)
        #         self.results.append(result)
        # else:
        #     # 기존 기본 평가 로직
        #     for criteria in self.evaluation_criteria:
        #         score, feedback = self._basic_evaluation(
        #             response, reference_answer, criteria
        #         )
        #         result = EvaluationResult(
        #             model_name=model_name,
        #             criteria_name=criteria.name,
        #             score=score,
        #             feedback=feedback,
        #             prompt=prompt,
        #             response=response
        #         )
        #         results.append(result)
        #         self.results.append(result)

        # # 평가 결과 저장 후 prompt별 가중 총점 계산
        # prompt_key = prompt
        # model_results_for_prompt = [r for r in self.results if r.prompt == prompt_key and r.model_name == model_name]
        
        # # 모든 criteria에 대한 점수를 모음
        # scores_by_criteria = {c.name: [] for c in self.evaluation_criteria}
        # for r in model_results_for_prompt:
        #      if r.criteria_name in scores_by_criteria:
        #          scores_by_criteria[r.criteria_name].append(r.score)

        # final_scores = []
        # weights = []
        # for c in self.evaluation_criteria:
        #     # 가장 최근 점수 사용
        #     score = scores_by_criteria[c.name][-1] if scores_by_criteria[c.name] else -1.0
        #     final_scores.append(score)
        #     weights.append(c.weight)

        # if sum(weights) > 0:
        #     valid_scores = [(s, w) for s, w in zip(final_scores, weights) if s != -1.0]
        #     if valid_scores:
        #         scores, ws = zip(*valid_scores)
        #         weighted_score = float(np.average(scores, weights=ws))
        #     else:
        #         weighted_score = -1.0
        # else:
        #     weighted_score = float(np.mean([s for s in final_scores if s != -1.0])) if any(s != -1.0 for s in final_scores) else -1.0

        # if prompt_key not in self.prompt_scores:
        #     self.prompt_scores[prompt_key] = {}
        # self.prompt_scores[prompt_key][model_name] = weighted_score