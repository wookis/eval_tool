import pandas as pd
from utils.logger import logger
import ast
import json

# 결측값 분석
def analyze_missing_values(df: pd.DataFrame):
    logger.info("\n=== 결측값 분석 ===")
    missing_values = df.isnull().sum()
    missing_values_percentage = (missing_values / len(df)) * 100
    missing_values_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_values_percentage
    })
    return missing_values_df


# 데이터 분포 상세 정보
def analyze_data_distribution(df: pd.DataFrame, column_name: str):
    print(f"\n=== {column_name} 데이터 분포 상세 정보 ===")
    return df[column_name].value_counts()


# eval 파일 생성
def create_eval_file(df: pd.DataFrame):
    logger.info("=== eval 파일 저장 ===")
    df_eval = pd.DataFrame({
        'id': df['id'].tolist(),
        'lr_answer_bronze_id' : df['lr_answer_bronze_id'].tolist(),
        'source_id': df['source_id'].tolist(),
        'messages': df['messages'].tolist(),
        'label_task': df['label_task'].tolist(), 
        'label_level': df['label_level'].tolist(), 
        'label_domain': df['label_domain'].tolist(),
        'answerer_llm_alias': df['answerer_llm_alias'].tolist(),
        'response': df['response'].tolist(),
        'content': df['content'].tolist(),
        'status_code': df['status_code'].tolist(),
        'ref': "",
        'req': "",
        'res': "",
        'eval_prompt': "",
        'eval_response': "",
        'eval_result': ""
    })

    for i, row in df_eval.iterrows():
        #list_message_as_dict = deserialize_messages_to_list_dict(row['messages'])
        # #list_message_as_dict = list(row['messages'])

        # if row['status_code'] == 200:
        #     content = row['content']
        #     list_message_as_dict.append({
        #         'role': ChatMessageRole.Assistant.value,
        #         'content': content
        #     })
        # else:
        #     raise Exception(f"status_code: {row['status_code']}. 응답 자가 없는 처리에 대해 고민 필요")
        try:
            messages_list = ast.literal_eval(row['messages'])

            response_str = row['response']
            response_list = json.loads(response_str)

            #print(type(response_list))
            #print(response_list['choices'][0]['message'])
            ref = ""
            req = messages_list[1]['content']
            res = response_list['choices'][0]['message']['content']

        except Exception as e:
            print(e)
            print(response_list)
            print(response_list['choices'][0])
            print(response_list['choices'][0]['message'])
            #res=""
            continue
       
            
        #res=""
       
        # #response_list 에서 content 추출
        # print(response_list['id'])
       
        # print(response_list['choices'][0]['message']['content'])
        #res = row['response']['choices'][0]['message']['content']
        if row['label_task'] is None or row['label_task'] == "":
            print("task_null =", req)
        #print(req)
        #print(res)
        #print(res)
        # print("원본", list_message_as_dict)
        # print("질문", list_list_message_as_dict[i][1]['content'])
        # print("답", list_list_message_as_dict[0][len(list_list_message_as_dict[0])-1]['content'])
        df_eval.loc[i, 'ref'] = ref 
        df_eval.loc[i, 'req'] = req #req
        # #     #print("저장", df_eval.loc[i, 'req'])
        df_eval.loc[i, 'res'] = res #res
    return df_eval


def dataset_validation(df_mode: str, df: pd.DataFrame):
    #response 파일 확인
    #print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    if df_mode == "df_eval":
        # 결측값 분석
        missing_values_df = analyze_missing_values(df)
        print(missing_values_df)
        print(missing_values_df[missing_values_df['Missing Count'] > 0])

        # 데이터 분포 상세 정보
        df_info = analyze_data_distribution(df, column_name="label_task")
        print(df_info)
        df_label_task_null = df['label_task'].isna().sum()
        df_response_null = df['response'].isna().sum()
        print(f"df_label_task_null: {df_label_task_null}")
        print(f"df_response_null: {df_response_null}")

        # label_task 결측 데이터 제외
        df = df[df['label_task'].notna()]

        # response 결측 데이터 제외
        df = df[df['label_task'].notna()]

        print(f"df_label_task_null: {df_label_task_null}")
        print(f"df_response_null: {df_response_null}")
    elif df_mode == "df_eval_result":
        # 평가 결과 분석
        df_eval_response_null = df['eval_response'].isna().sum()
        df_eval_result_null = df['eval_result'].isna().sum()
        #df['eval_result'] 가 null이 아닌 데이터 개수
        df_eval_result_cnt = df[df['eval_result'].notna()].shape[0]
        #print(f"df_eval_response_null: {df_eval_response_null}")
        print(f"평가 데이터 : {df_eval_result_cnt}")
        print(f"미평가 데이터 : {df_eval_result_null}")
        
        # 데이터 분포 상세 정보
        #df_info = analyze_data_distribution(df, column_name="label_task")

    return df