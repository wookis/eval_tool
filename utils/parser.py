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
    #print(df["answerer_llm_alias"].value_counts())
    print("label_task 없는 데이터 수", df[df['label_task'].isna()].shape)
    df = df[df['label_task'].notna()]
    #print("label_task 없는 데이터 제거 후", df.shape)
    #print(df["answerer_llm_alias"].value_counts())

    #answerer_llm_alias 결측값 확인 및 제거
    #print("answerer_llm_alias 없는 데이터 제거 전", df.shape)
    #print(df["answerer_llm_alias"].value_counts())
    print("answerer_llm_alias 없는 데이터 수", df[df['answerer_llm_alias'].isna()].shape)
    df = df[df['answerer_llm_alias'].notna()]
    #print("answerer_llm_alias 없는 데이터 제거 후", df.shape)
    #print(df["answerer_llm_alias"].value_counts())

    #eval_result 결측값 확인 및 제거
    #print("eval_result 없는 데이터 제거 전", df.shape)
    #print(df["answerer_llm_alias"].value_counts())
    print("eval_result 없는 데이터 수", df[df['eval_result'].isna()].shape)
    df = df[df['eval_result'].notna()]
    #print("eval_result 없는 데이터 제거 후", df.shape)
    #print(df["answerer_llm_alias"].value_counts())

    #df_filtered answerer_llm_alias 결측값 제거
    #print("LLM 정보 없는 데이터 제거 전", df_filtered.shape)
    print("LLM 정보 없는 데이터 수", df[df['answerer_llm_alias'].isna()].shape)
    df = df[df['answerer_llm_alias'].notna()]
    #print("LLM 정보 없는 데이터 제거 후", df_filtered.shape)

    print("결측치 데이터 제거 후", df.shape)
    print("answerer_llm_alias unique", df['answerer_llm_alias'].unique())

   
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