import pandas as pd
from utils.logger import logger
import ast
import json
import streamlit as st
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import itertools
from tqdm import tqdm
import re
from matplotlib.colors import LinearSegmentedColormap




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
        logger.info(missing_values_df)
        logger.info(missing_values_df[missing_values_df['Missing Count'] > 0])

        # 데이터 분포 상세 정보
        df_info = analyze_data_distribution(df, column_name="label_task")
        logger.info(df_info)
        df_label_task_null = df['label_task'].isna().sum()
        df_response_null = df['response'].isna().sum()
        logger.info(f"df_label_task_null: {df_label_task_null}")
        logger.info(f"df_response_null: {df_response_null}")

        # label_task 결측 데이터 제외
        df = df[df['label_task'].notna()]
        # response 결측 데이터 제외
        df = df[df['response'].notna()]

        #logger.info(f"df_label_task_null: {df_label_task_null}")
        #logger.info(f"df_response_null: {df_response_null}")
    elif df_mode == "df_eval_result":
        # 평가 결과 분석
        #df_eval_response_null = df['eval_response'].isna().sum()
        df_eval_result_null = df['eval_result'].isna().sum()
        #df['eval_result'] 가 null이 아닌 데이터 개수
        df_eval_result_cnt = df[df['eval_result'].notna()].shape[0]
        logger.info(f"전체 데이터 : {len(df)}")
        logger.info(f"평가 데이터 : {df_eval_result_cnt}")
        logger.info(f"미평가 데이터 : {df_eval_result_null}")
        
        # 데이터 분포 상세 정보
        #df_info = analyze_data_distribution(df, column_name="label_task")

    return df

def filter_efficient_llm(list_dict_list_llm_id_and_quality_and_cost):
    """
    list_dict_list_llm_id_and_quality_and_cost: [{"llm_ids": list[int], "quality": float, "cost": float}, ...]
    Pareto front 계산 후 원본 순서 유지
    """
    # 1️⃣ 후보별 orig_idx 추가 (llm_id 기준으로 순서 보존)
    # 2️⃣ cost 오름차순, quality 내림차순으로 정렬
    list_dict_list_llm_id_and_quality_and_cost_sorted = sorted(list_dict_list_llm_id_and_quality_and_cost, key=lambda x: (round(x["cost"], 7), -round(x["quality"], 7)))

    # 3️⃣ Pareto front 계산
    efficient = []
    max_quality = -np.inf
    for dict_list_llm_id_and_quality_and_cost_sorted in list_dict_list_llm_id_and_quality_and_cost_sorted:
        if dict_list_llm_id_and_quality_and_cost_sorted["quality"] > max_quality:
            efficient.append(dict_list_llm_id_and_quality_and_cost_sorted)
            max_quality = dict_list_llm_id_and_quality_and_cost_sorted["quality"]

    return efficient


def get_efficient_tld_by_step_by_step(list_list_dict_list_llm_id_and_quality_and_cost):
    list_dict_list_llm_id_and_quality_and_cost_efficient = list_list_dict_list_llm_id_and_quality_and_cost[0]

    for index_target in tqdm(range(1,len(list_list_dict_list_llm_id_and_quality_and_cost))):

        list_dict_list_llm_id_and_quality_and_cost_candidate = [
            {
                "list_llm_id": a["list_llm_id"] + b["list_llm_id"],
                "cost": a["cost"] + b["cost"],
                "quality": a["quality"] + b["quality"],
            } for a, b in itertools.product(list_dict_list_llm_id_and_quality_and_cost_efficient, list_list_dict_list_llm_id_and_quality_and_cost[index_target])
        ]
            
        list_dict_list_llm_id_and_quality_and_cost_efficient = filter_efficient_llm(list_dict_list_llm_id_and_quality_and_cost_candidate)

        #print(f"length of result: {len(list_dict_list_llm_id_and_quality_and_cost_efficient)}")
        #print(f"length of list_llm_id: {len(list_dict_list_llm_id_and_quality_and_cost_efficient[0]['list_llm_id'])}")
    return list_dict_list_llm_id_and_quality_and_cost_efficient

# 자연 정렬 함수
def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def pivot_df_as_tld2tld_as_table(df_as_tld, values=None, aggfunc=None):
    ## 1. index, columns 생성    
    list_task_label = [f"T{i}" for i in range(1, 13)]
    list_domain_label = [f"D{i}" for i in range(1, 11)]
    list_level_label = [f"L{i}" for i in range(1, 4)]

    list_labels = []
    for t in list_task_label:
            for d in list_domain_label:
                for l in list_level_label:
                    list_labels.append({
                        "T": t,
                        "D": d,
                        "L": l,
                    })
    df_tld_template = pd.DataFrame(list_labels)
    df_tld_template['count'] = 0.0
    df_pivot = df_tld_template.pivot_table(index=['T', 'L'], columns='D', values='count')  # quality, cost, count 등 아무거나 상관없음 (NaN 남김)
    t_vals = [t for t, l in df_pivot.index]
    sorted_idx = sorted(range(len(t_vals)), key=lambda i: natural_key(t_vals[i]))
    sorted_columns = sorted(df_pivot.columns, key=lambda x: natural_key(x))
    df_pivot_sorted = df_pivot.iloc[sorted_idx][sorted_columns]
    ## 2. 실 데이터 reindex + columns 정렬
    df_as_table_pivot = df_as_tld.pivot_table(index=['T', 'L'], columns='D', values=values, aggfunc=aggfunc)  # quality, cost, count 등 아무거나 상관없음 (NaN 남김)

    ## 250917 dusanbaek 수정함
    # return df_as_table_pivot.reindex(df_pivot_sorted.index)[sorted_columns]
    df_as_table_pivot = df_as_table_pivot.reindex(df_pivot_sorted.index)
    for column_to_sort in sorted_columns:
        if column_to_sort not in df_as_table_pivot.columns:
            df_as_table_pivot[column_to_sort] = np.NaN
    return df_as_table_pivot[sorted_columns]

def get_TLDMapData(df_filtered: pd.DataFrame):
    dict_llm_name2df_records = {}

    for answerer_llm_alias in df_filtered['answerer_llm_alias'].unique().tolist():
        dict_llm_name2df_records[answerer_llm_alias] = df_filtered[df_filtered['answerer_llm_alias'] == answerer_llm_alias]
        dict_llm_name2df_records[answerer_llm_alias].reset_index(drop=True, inplace=True)

    list_task_label = [f"T{i}" for i in range(1, 13)]
    list_domain_label = [f"D{i}" for i in range(1, 11)]
    list_level_label = [f"L{i}" for i in range(1, 4)]
    list_llm_name = list(dict_llm_name2df_records.keys())

    dict_llm_name2df = {}    

    for llm_name, df in dict_llm_name2df_records.items():
        # print(f"{df.head(20)=}")
        list_labels = []
        for t in list_task_label:
            for d in list_domain_label:
                for l in list_level_label:
                    list_labels.append({
                        "T": t,
                        "D": d,
                        "L": l,
                    })
        df_tld = pd.DataFrame(list_labels)
        df_tld['count'] = 0.0
        df_tld['quality'] = 0.0
        df_tld['cost'] = 0.0

        #df_tld.set_index(['T', 'D', 'L'], inplace=True)

        # print("22222 : ", llm_name)
        
        # print(len(df))

        for index, row in tqdm(df.iterrows()):
            
            T = row['label_task']
            D = row['label_domain']
            L = row['label_level']
            cost = row['cost']
            quality = row['quality']
            
            normalization_factor = 1 / (len(T) * len(D) * len(L))                
            # for t, d, l in itertools.product(T, D, L):
            #     key = (t, d, l)
            #     if key in df_tld.index:
            #         df_tld.loc[key, 'count'] += normalization_factor
            #         df_tld.loc[key, 'cost'] += cost * normalization_factor
            #         df_tld.loc[key, 'quality'] += quality * normalization_factor
                #print(f"{key}")
                #print(f'{df_tld.index=}')

            for t_to_count, d_to_count, l_to_count in itertools.product(T, D, L):
                mask = (
                    (df_tld['T'] == t_to_count) &
                    (df_tld['D'] == d_to_count) &
                    (df_tld['L'] == l_to_count)
                )
                df_tld.loc[mask, 'count'] += normalization_factor
                df_tld.loc[mask, 'cost'] += cost * normalization_factor
                df_tld.loc[mask, 'quality'] += quality * normalization_factor
        
        dict_llm_name2df[llm_name] = df_tld
        print(f'{llm_name}: {len(dict_llm_name2df[llm_name])}')

        # print("33333 : ", llm_name)
        # print(f"{df_tld.head(20)=}")

        #print(f'{llm_name}: {len(dict_llm_name2df[llm_name])}')

    
    #ßdict_llm_name2df['gpt-4.1'].head(2)
    # dict_llm_name2df는 딕셔너리이므로 각 모델별 DataFrame을 개별적으로 저장
    print(f"dict_llm_name2df contains {len(dict_llm_name2df)} models")
    
    # 각 모델별 DataFrame을 개별 CSV 파일로 저장 (선택사항)
    # for model_name, df_model in dict_llm_name2df.items():
    #     filename = f'dict_llm_name2df_{model_name.replace("-", "_")}.csv'
    #     df_model.to_csv(filename, index=False)
    #     print(f"Saved {filename}")

    # 모든 DataFrame을 하나로 합쳐서 저장
    all_data = []
    for model_name, df_model in dict_llm_name2df.items():
        df_with_model = df_model.copy()
        df_with_model['model_name'] = model_name
        all_data.append(df_with_model)
        
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv('all_models_data.csv', index=False)


    dict_llm_id2llm_name = {
    index: llm_name for index, llm_name in enumerate(dict_llm_name2df.keys())
    }
    print(f"{dict_llm_id2llm_name[0]=}")
    dict_llm_id2df = {index: df for index, df in enumerate(dict_llm_name2df.values())}
    dict_llm_id2df[0].head(2)

    ## count가 0이 아닌 것만 모음

    def get_row_from_df_tld(df_tld, T, D, L):
        return df_tld[(df_tld['T'] == T) & (df_tld['D'] == D) & (df_tld['L'] == L)]

    df_all_candidate_tld = pd.DataFrame(columns=['T', 'D', 'L', 'list_dict_list_llm_ids_and_quality_and_cost_efficient', 'count'])

    df_sample_tld = dict_llm_id2df[0]

    for index, current_target_row in tqdm(df_sample_tld[df_sample_tld['count'].ne(0)].iterrows()):
        T = current_target_row['T']
        D = current_target_row['D']
        L = current_target_row['L']
        count = current_target_row['count']

        list_dict_list_llm_id_and_quality_and_cost = []

        for candidate_llm_id, df_tld_to_check in dict_llm_id2df.items():
            row_to_check = get_row_from_df_tld(df_tld_to_check, T, D, L)
            candidate_quality = row_to_check['quality'].values[0]
            candidate_cost = row_to_check['cost'].values[0]
            list_dict_list_llm_id_and_quality_and_cost.append({'list_llm_id': [candidate_llm_id], 'quality': candidate_quality, 'cost': candidate_cost})

        list_dict_list_llm_id_and_quality_and_cost_efficient = filter_efficient_llm(list_dict_list_llm_id_and_quality_and_cost)
        # print(T,D,L)
        # print(list_dict_list_llm_id_and_quality_and_cost_efficient)
        if len(list_dict_list_llm_id_and_quality_and_cost_efficient) == 0:
            print(T,D,L)
            print(list_dict_list_llm_id_and_quality_and_cost_efficient)
            break
        df_all_candidate_tld.loc[len(df_all_candidate_tld)] = [T, D, L, list_dict_list_llm_id_and_quality_and_cost_efficient, count]
    df_all_candidate_tld.head(2)

    ## count가 0개가 아닌것만 처음에 모였고,
    ## TLD 조합 내 최적 LLM이 1개인 것들을 별도로 모음
    df_already_efficient_tld = df_all_candidate_tld[df_all_candidate_tld['list_dict_list_llm_ids_and_quality_and_cost_efficient'].map(len) == 1].reset_index(drop=True)
    print("df_already_efficient_tld.shape : ", df_already_efficient_tld.shape)
    print("dddddddddddddd")

    ## TLD 조합 내 최적 LLM이 1개인 것들의 quality, cost를 누적 (어떠한 조합에서도 누적된 quality, cost는 기본이됨)
    total_quality_of_efficient_tld = 0
    total_cost_of_efficient_tld = 0
    for _, df_already_efficient_tld_row in df_already_efficient_tld.iterrows():
        total_quality_of_efficient_tld += df_already_efficient_tld_row['list_dict_list_llm_ids_and_quality_and_cost_efficient'][0]['quality']
        total_cost_of_efficient_tld += df_already_efficient_tld_row['list_dict_list_llm_ids_and_quality_and_cost_efficient'][0]['cost']
    
    ## TLD 조합 내 최적 LLM이 2개 이상인 것들을 추림
    ## 이것들을 가지고 최적 LLM을 찾아야 함
    df_to_find_efficient_tld = df_all_candidate_tld[df_all_candidate_tld['list_dict_list_llm_ids_and_quality_and_cost_efficient'].map(len) >= 2].reset_index(drop=True)
    print("df_to_find_efficient_tld.shape : ", df_to_find_efficient_tld.shape)

    df_to_find_efficient_tld['list_dict_list_llm_ids_and_quality_and_cost_efficient'].to_list()

    ## TLD 조합 내 최적 LLM이 2개 이상인 것들을 입력해서
    ## 최적을 찾음
    efficient_tld = get_efficient_tld_by_step_by_step(df_to_find_efficient_tld['list_dict_list_llm_ids_and_quality_and_cost_efficient'].to_list())

    ## 2개 이상이었던 efficient_tld에 1개짜리 값 추가
    for dict_efficient_tld in efficient_tld:
        dict_efficient_tld['cost'] += total_cost_of_efficient_tld
        dict_efficient_tld['quality'] += total_quality_of_efficient_tld


    ## 예시로 1개만 만들어 보기 위한 샘플 데이터 확보
    pdf_efficient_tld_sample = efficient_tld[-1]
    pdf_efficient_tld_sample['llm_name'] = [dict_llm_id2llm_name[llm_id] for llm_id in pdf_efficient_tld_sample['list_llm_id']]
    pdf_merged_sample = pd.concat([df_to_find_efficient_tld, pd.DataFrame(pdf_efficient_tld_sample)], axis=1)
    print("pdf+merge", pdf_merged_sample.head(2))

    ## TLD Table 만들기 위해 pivot table 만듬. 이 때의 값은 위 샘플
    pdf_merged_sample_llm_name_as_tld_table = pivot_df_as_tld2tld_as_table(pdf_merged_sample, values='llm_name', aggfunc=lambda x: '\n'.join(x))
    #pdf_merged_sample_llm_name_as_tld_table.to_csv('./processed-data/highest-quality-tldc-all.csv', encoding='utf-8-sig', index=True)
    print(pdf_merged_sample_llm_name_as_tld_table.head(2))
    print("pdf_merged_sample_llm_name_as_tld_tablepdf_merged_sample_llm_n")
    
    
    ## TLD Table 만들기 위해 pivot table 만듬. 이 때의 값은 count이며, heatmap과 같은 색조를 넣기 위해서 사용됨
    df_efficient_tld_num_llm_as_tld_table = pivot_df_as_tld2tld_as_table(pdf_merged_sample, values='count', aggfunc=sum)
    print(df_efficient_tld_num_llm_as_tld_table.head(2))
    print("df_efficient_tld_num_llm_as_tld_tabledf_efficient_tld_num_llm_as_tld_tabledf_efficient_tld_num_llm_as_tld_table")

    ChartView.plot_tld_table(df_efficient_tld_num_llm_as_tld_table, pdf_merged_sample_llm_name_as_tld_table, 'highest_quality')


    return efficient_tld
    
class EDA:
    def __init__(self):
        pass

    def analyze_missing_values(self, df: pd.DataFrame):
        return analyze_missing_values(df)

    def analyze_data_distribution(self, df: pd.DataFrame, column_name: str):
        return analyze_data_distribution(df, column_name)

    def create_eval_file(self, df: pd.DataFrame):
        return create_eval_file(df)

    def dataset_validation(self, df_mode: str, df: pd.DataFrame):
        return dataset_validation(df_mode, df)

    def TLDMapData(df_filtered: pd.DataFrame):
        return get_TLDMapData(df_filtered)


class ChartView:
    def __init__(self):
        pass

    # def create_chart(self, df: pd.DataFrame):
    #     return create_chart(df)

    # def create_chart_distribution(self, df: pd.DataFrame, column_name: str):
    #     return create_chart_distribution(df, column_name)

    @staticmethod
    def DataEDA(df: pd.DataFrame):

        print(df.columns)
        st.title(f"평가 대상 데이터 분포 차트")
        st.write("평가 대상 데이터 구성에 대한 차트정보 입니다.")
        # 1. 카테고리별 분포 차트
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')

        #model_name 컬럼이름 변경
        df.rename(columns={'model_name': 'answerer_llm_alias'}, inplace=True)

        print(df.columns)

        # 0-0. answerer_llm_alias 분포
        if 'answerer_llm_alias' in df.columns:
            alias_counts = df['answerer_llm_alias'].value_counts()
            axes[0, 0].bar(range(len(alias_counts)), alias_counts.values, color='red')
            axes[0, 0].set_title('Answerer LLM Alias Distribution')
            #axes[0, 0].set_xlabel('LLM Alias')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_xticks(range(len(alias_counts)))
            axes[0, 0].set_xticklabels(alias_counts.index, rotation=45)

            for i, v in enumerate(alias_counts.values):
                axes[0, 0].text(i, v, str(v), ha='center', va='bottom')

        # 0-1. source_id 분포
        if 'source_id' in df.columns:
            source_id_counts = df['source_id'].value_counts()
            axes[0, 1].bar(range(len(source_id_counts)), source_id_counts.values, color='blue')
            axes[0, 1].set_title('source_id Distribution')
            #axes[0, 1].set_xlabel('LLM Alias')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_xticks(range(len(source_id_counts)))
            axes[0, 1].set_xticklabels(source_id_counts.index, rotation=45)

            for i, v in enumerate(source_id_counts.values):
                axes[0, 1].text(i, v, str(v), ha='center', va='bottom')

        # 1-0. label_task 분포 (상위 10개만)
        if 'label_task' in df.columns:
            task_counts = df['label_task'].value_counts().head(10)
            axes[1, 0].bar(range(len(task_counts)), task_counts.values, color='orange')
            axes[1, 0].set_title('Label Task Distribution (Top 10)')
            #axes[1, 0].set_xlabel('Task')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(len(task_counts)))
            axes[1, 0].set_xticklabels(task_counts.index, rotation=45)
            for i, v in enumerate(task_counts.values):
                axes[1, 0].text(i, v, str(v), ha='center', va='bottom')

        # 1-1. label_task 분포 (하위 10개만)
        if 'label_task' in df.columns:
            task_counts = df['label_task'].value_counts().tail(10)
            axes[1, 1].bar(range(len(task_counts)), task_counts.values, color='orange')
            axes[1, 1].set_title('Label Task Distribution (Top 10)')
            #axes[1, 1].set_xlabel('Task')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_xticks(range(len(task_counts)))
            axes[1, 1].set_xticklabels(task_counts.index, rotation=45)
            for i, v in enumerate(task_counts.values):
                axes[1, 1].text(i, v, str(v), ha='center', va='bottom')

        # 2-0. label_domain 분포
        if 'label_domain' in df.columns:
            domain_counts = df['label_domain'].value_counts().head(10)
            axes[2, 0].bar(range(len(domain_counts)), domain_counts.values, color='skyblue')
            axes[2, 0].set_title('Label Domain Distribution (Top 10)')
            #axes[2, 0].set_xlabel('Domain')
            axes[2, 0].set_ylabel('Count')
            axes[2, 0].set_xticks(range(len(domain_counts)))
            axes[2, 0].set_xticklabels(domain_counts.index, rotation=45)
            for i, v in enumerate(domain_counts.values):
                axes[2, 0].text(i, v, str(v), ha='center', va='bottom')

        # 2-1. label_level 분포
        if 'label_level' in df.columns:
            level_counts = df['label_level'].value_counts()
            axes[2, 1].bar(range(len(level_counts)), level_counts.values, color='lightgreen')
            axes[2, 1].set_title('Label Level Distribution')
            #axes[2, 1].set_xlabel('Level')
            axes[2, 1].set_ylabel('Count')
            axes[2, 1].set_xticks(range(len(level_counts)))
            axes[2, 1].set_xticklabels(level_counts.index, rotation=45)
            for i, v in enumerate(level_counts.values):
                axes[2, 1].text(i, v, str(v), ha='center', va='bottom')

        plt.tight_layout()
        #plt.show()
        st.pyplot(fig)

        if 'answerer_llm_alias' in df.columns:
            print(f"\nAnswerer LLM Alias 분포:")
            print(df['answerer_llm_alias'].value_counts())

        if 'label_domain' in df.columns:
            print(f"\nLabel Domain 분포:")
            print(df['label_domain'].value_counts())


    @staticmethod
    def ModelEvalDist(df_filtered: pd.DataFrame):
        st.title(f"모델별 평가 결과 분포 차트")
        st.write("모델별 평가 결과 분포 차트 입니다.")
        
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        # 스타일 설정
        plt.style.use('default')
        sns.set_palette("Set2")

        # 1. 서브플롯으로 각 모델별 히스토그램 분리
        unique_models = df_filtered["answerer_llm_alias"].unique()
        print(df_filtered.describe())
        print(f"{unique_models=}")

        #sys.exit()
        n_models = len(unique_models)

        # 서브플롯 레이아웃 계산
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        
        # axes를 항상 2차원 배열로 만들기
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        print(f"총 {n_models}개 모델: {list(unique_models)}")

        for i, model in enumerate(unique_models):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            subset = df_filtered[df_filtered["answerer_llm_alias"] == model]
            
            # 데이터 통계
            mean_val = subset["quality"].mean()
            std_val = subset["quality"].std()
            count = len(subset)
            
            print(f"{model}: 평균={mean_val:.3f}, 표준편차={std_val:.3f}, 개수={count}")
            
            # 히스토그램 그리기
            ax.hist(subset["quality"].values, 
                    bins=30, 
                    alpha=0.8, 
                    color=f'C{i}',
                    edgecolor='black',
                    linewidth=0.5)
            
            # 평균선 추가
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            
            # 차트 꾸미기
            ax.set_title(f'{model}\n(n={count})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Quality Score', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)

        # 빈 서브플롯 숨기기
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)

        plt.tight_layout()
        #plt.show()
        st.pyplot(fig)


        # 4. 통계 요약 테이블
        print("\n=== 모델별 통계 요약 ===")
        summary_stats = df_filtered.groupby('answerer_llm_alias')['quality'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        summary_stats.columns = ['Count', 'Mean', 'Std', 'Min', 'Max', 'Median']
        print(summary_stats)
        st.write(summary_stats)
    
    def ModelScoreDist(df_filtered: pd.DataFrame):
        desc = df_filtered.groupby("answerer_llm_alias")["quality"].describe()
        print(desc)
        # 5%, 95% 백분위수 추가 계산
        percentiles = df_filtered.groupby("answerer_llm_alias")["quality"].quantile([0.05, 0.95])
        
        # 필요한 column만 추출하고 백분위수 추가
        summary = desc[["mean", "25%", "50%", "75%", "max"]].copy()
        
        # 5%, 95% 백분위수를 별도 컬럼으로 추가
        summary["5%"] = percentiles.xs(0.05, level=1)
        summary["95%"] = percentiles.xs(0.95, level=1)
        
        # 컬럼 순서 재정렬
        summary = summary[["mean", "5%", "25%", "50%", "75%", "95%", "max"]]
        print(summary.head())  # 확인
        
        # Streamlit에서 표시
        st.write("## 모델별 품질 스코어 통계")
        st.dataframe(summary.round(3))

        summary.T.plot(kind="bar", figsize=(10,6))
        plt.title("Quality Statistics by Model")
        plt.ylabel("Value")
        plt.xlabel("Statistic")
        plt.legend(title="Model", loc='lower right')
        st.pyplot(plt)

    def ModelCostDist(df_filtered: pd.DataFrame):
        desc = df_filtered.groupby("answerer_llm_alias")["cost"].describe()
        print(desc)
        # 5%, 95% 백분위수 추가 계산
        percentiles = df_filtered.groupby("answerer_llm_alias")["cost"].quantile([0.05, 0.95])
        
        # 필요한 column만 추출하고 백분위수 추가
        summary = desc[["mean", "25%", "50%", "75%", "max"]].copy()
        
        # 5%, 95% 백분위수를 별도 컬럼으로 추가
        summary["5%"] = percentiles.xs(0.05, level=1)
        summary["95%"] = percentiles.xs(0.95, level=1)
        
        # 컬럼 순서 재정렬
        summary = summary[["mean", "5%", "25%", "50%", "75%", "95%", "max"]]
        print(summary.head())  # 확인
        
        # Streamlit에서 표시
        st.write("## 모델별 품질 스코어 통계")
        st.dataframe(summary.round(3))

        summary.T.plot(kind="bar", figsize=(10,6))
        plt.title("Cost Statistics by Model")
        plt.ylabel("Value")
        plt.xlabel("Statistic")
        plt.legend(title="Model", loc='lower right')
        st.pyplot(plt)

    def plot_tld_table(df_for_heatmap, df_for_anno, title, annot_fontsize=10):

        task_guideline = '''
        ### Task Labels
        - T1: Translation — 자연어 간 번역
        - T2: Summarization — 문장이나 문서의 내용을 짧게 요약
        - T3: Math Problem Solving — 수식 기반 문제 풀이 관련 요청, 수식을 세워 해결해야 하는 문제
        - T4: Code Tasks — 프로그래밍 코드 생성, 수정, 설명, 디버깅 전반
        - T5: Text Editing — 문장의 본래 의미 보존하며 문장 고치기, 문법 수정, 문체 정돈, 공손화
        - T6: Image Generation — 이미지 생성
        - T7: Creative Writing — 광고 문구/시/소설 등 창의적이고 감성 중심의 텍스트 생성, 문장 스타일 또는 감정의 톤 변환
        - T8: Planning/Reasoning — 선택/판단/의사결정에 필요한 전략/제안, 의견/조언, 비교/분석 요청, 원인/관계 추론, 가정(if)/조건부분기
        *예: "퇴사 후 프리랜서로 전향할 수 있는 방법을 알려줘", "A는 싸고 B는 가깝다면 뭘 선택하는게 좋을까?", "왜 금리가 오르면 환율이 떨어질까?"*
        - T9: Classification — 텍스트에 대한 카테고리 판별
        - T10: Factoid QA — 사실 기반 질의, 정의 설명, 단일 정보 조회, 명확한 정답 존재
        *예: "지구의 자전 속도는 얼마나 되나요?"*
        - T11: Chat — 감정 교류, 공감, 위로, 응답 반응 중심의 대화, 운세
        - T12: Unknown — 어떤 Task로도 분류할 수 없는 요청 (모호한 요청, 해석불가한 입력 등)


        ##### 숫자 포함 but 역사적 사실 질문  
        {"messages":[{"role":"system","content":""},{"role":"user","content":"1945년에 일본이 항복했는데, 그때 한국은 몇 년간 식민지 상태였던 거야?"}]}
        T9


        ##### 번역 + 공손한 표현 요청  
        {"messages":[{"role":"system","content":""},{"role":"user","content":"이 문장을 영어로 번역하고 더 공손하게 바꿔줘"}]}
        T1,T5


        ##### 수식 계산 + 금융 판단  
        {"messages":[{"role":"system","content":""},{"role":"user","content":"3% 복리로 5년 동안 투자하면 1천만 원은 얼마야? 단리일 때랑 비교도 해줘"}]}
        T3,T8
        '''

        dict_task_label2short_explanation = {
            "T1": "Translation",
            "T2": "Summarization",
            "T3": "Math Problem Solving",
            "T4": "Code Tasks",
            "T5": "Text Editing",
            "T6": "Image Generation",
            "T7": "Creative Writing",
            "T8": "Planning/Reasoning",
            "T9": "Classification",
            "T10": "Factoid QA",
            "T11": "Chat",
            "T12": "Unknown"
        }

        dict_task_label2full_explanation = {
            "T1": "Translation - 자연어 간 번역",
            "T2": "Summarization - 문장이나 문서의 내용을 짧게 요약",
            "T3": "Math Problem Solving - 수식 기반 문제 풀이 관련 요청, 수식을 세워 해결해야 하는 문제",
            "T4": "Code Tasks - 프로그래밍 코드 생성, 수정, 설명, 디버깅 전반",
            "T5": "Text Editing - 문장의 본래 의미 보존하며 문장 고치기, 문법 수정, 문체 정돈, 공손화",
            "T6": "Image Generation - 이미지 생성",
            "T7": "Creative Writing - 광고 문구/시/소설 등 창의적이고 감성 중심의 텍스트 생성, 문장 스타일 또는 감정의 톤 변환",
            "T8": "Planning/Reasoning - 선택/판단/의사결정에 필요한 전략/제안, 의견/조언, 비교/분석 요청, 원인/관계 추론, 가정(if)/조건부분기",
            "T9": "Classification - 텍스트에 대한 카테고리 판별",
            "T10": "Factoid QA - 사실 기반 질의, 정의 설명, 단일 정보 조회, 명확한 정답 존재",
            "T11": "Chat - 감정 교류, 공감, 위로, 응답 반응 중심의 대화, 운세",
            "T12": "Unknown - 어떤 Task로도 분류할 수 없는 요청 (모호한 요청, 해석불가한 입력 등)"
        }

        domain_guideline = '''
        ### Domain Labels
        - D1: Mathematics/Statistics — 수학, 통계
        - D2: Finance/Economics — 금융, 투자, 경제학
        - D3: Computing/Programming — 컴퓨터 과학, 프로그래밍, 소프트웨어 및 시스템 개발, 인공지능 기술
        - D4: Health/Medicine — 건강, 질병, 치료, 의학 정보
        - D5: Science/Technology — 과학, 공학, 기술 일반
        - D6: Law — 법률, 계약
        - D7: Politics — 정치 제도, 정부 정책, 선거, 정당, 정치사상, 외교 정책 등 정치 전반
        *예: "대통령 탄핵 요건은?", "윤석열 정부의 부동산 정책은?"*
        - D8: Humanities/Society/Culture — 인문학, 사회학, 역사, 철학, 예술, 문화 전반
        - D9: Daily Life — 일상 활동, 소비, 취미, 일정, 일반적인 사무 업무
        *예: "서울 근처에서 이번 주말에 볼만한 전시 뭐 있어?", "출장비 정산 어떻게 해?"*
        - D10: Uncategorized — 무의미하거나 해석 불가한 문자열,  도메인을 판단하기 어려운 요청, 비지식성 입력


        ##### 숫자 포함 but 역사적 사실 질문  
        {"messages":[{"role":"system","content":""},{"role":"user","content":"1945년에 일본이 항복했는데, 그때 한국은 몇 년간 식민지 상태였던 거야?"}]}
        D8

        ##### 법률 용어 없음 but 법률 문제  
        {"messages":[{"role":"system","content":""},{"role":"user","content":"퇴사했는데 회사가 연차 수당을 안 줘. 받을 수 있는 방법이 있을까?"}]}
        D6

        ##### 금융 지식 포함 수학계산  
        {"messages":[{"role":"system","content":""},{"role":"user","content":"매년 6% 수익을 주는 ETF에 3년간 매달 50만 원씩 정기적립하면 최종 수익은 얼마야? 같은 조건으로 예금에 넣었을 때보다 얼마나 차이 나?"}]}
        D1,D2

        ##### 과학 지식 포함 수식  
        {"messages":[{"role":"system","content":""},{"role":"user","content":"질량 10kg짜리 물체를 지면에서 5미터 위로 들어올리려면 얼마나 많은 에너지가 필요한가요?"}]}
        D1,D5
        '''

        dict_domain_label2short_explanation = {
            "D1": "Mathematics/Statistics",
            "D2": "Finance/Economics",
            "D3": "Computing/Programming",
            "D4": "Health/Medicine",
            "D5": "Science/Technology",
            "D6": "Law",
            "D7": "Politics",
            "D8": "Humanities/Society/Culture",
            "D9": "Daily Life",
            "D10": "Uncategorized"
        }

        dict_domain_label2full_explanation = {
            "D1": "Mathematics/Statistics - 수학, 통계",
            "D2": "Finance/Economics - 금융, 투자, 경제학",
            "D3": "Computing/Programming - 컴퓨터 과학, 프로그래밍, 소프트웨어 및 시스템 개발, 인공지능 기술",
            "D4": "Health/Medicine - 건강, 질병, 치료, 의학 정보",
            "D5": "Science/Technology - 과학, 공학, 기술 일반",
            "D6": "Law - 법률, 계약",
            "D7": "Politics - 정치 제도, 정부 정책, 선거, 정당, 정치사상, 외교 정책 등 정치 전반",
            "D8": "Humanities/Society/Culture - 인문학, 사회학, 역사, 철학, 예술, 문화 전반",
            "D9": "Daily Life - 일상 활동, 소비, 취미, 일정, 일반적인 사무 업무",
            "D10": "Uncategorized - 무의미하거나 해석 불가한 문자열, 도메인을 판단하기 어려운 요청"
        }

        level_guideline = '''
        ### Level Labels
        - L1: Easy — 오직 일상적이고 쉬운 어휘, Factoid, 단순 정의형 질의, reasoning 불필요
        *예: "다음 문장을 영어로 번역해줘: 집에 가고 싶어", "서울은 어느 나라의 수도인가요?"*
        - L2: Intermediate - 고등학교 교과과정 수준 이하의 어휘, 판단이 필요없는 단순 개념적 비교, 요청 처리를 위한 맥락 부족
        *예: "산화 환원 반응이란 무엇이고, 일상 생활에서 볼 수 있는 예시는 어떤 것이 있나요?", "수열과 급수의 차이점", "이 글을 읽고 요약해줘"*
        - L3: Complex - 대학교 전공과목 수준 이상의 어휘, 선택/판단/의사결정에 필요한 전략/제안, 의견/조언, 비교/분석 요청, 원인/관계 추론, 가정(if)/조건부분기
        *예: "다음의 'Neuroradiology 연구를 위한 VX2 세포를 이용한 토끼 뇌종양 모델 제작과 MRI를 이용한 검증' 논문 초록을 한문장으로 요약해줘.
        '''

        dict_level_label2short_explanation = {
            "L1": "Easy",
            "L2": "Intermediate",
            "L3": "Complex"
        }

        dict_level_label2full_explanation = {
            "L1": "Easy - 오직 일상적이고 쉬운 어휘, Factoid, 단순 정의형 질의, reasoning 불필요",
            "L2": "Intermediate - 고등학교 교과과정 수준 이하의 어휘, 판단이 필요없는 단순 개념적 비교, 요청 처리를 위한 맥락 부족",
            "L3": "Complex - 대학교 전공과목 수준 이상의 어휘, 선택/판단/의사결정에 필요한 전략/제안, 의견/조언, 비교/분석 요청, 원인/관계 추론, 가정(if)/조건부분기"
        }

        print("df_for_heatmapdf_for_heatmapdf_for_heatmapdf_for_heatmap")
        
        TUPLE_FIGSIZE = (30,15)
        # 0은 흰색, 그 이후는 YlOrRd 그대로 쓰기
        orig_cmap = sns.color_palette("YlOrRd", as_cmap=True)
        colors = [(1, 1, 1)] + [orig_cmap(i) for i in np.linspace(0, 1, 255)[1:]]  # 첫 색만 흰색으로
        custom_cmap = LinearSegmentedColormap.from_list("custom_YlOrRd", colors, N=256)

        # T, L 분리
        T_vals = [t for t, l in df_for_heatmap.index]
        L_vals = [l for t, l in df_for_heatmap.index]

        # 컬러맵: 0은 흰색, NaN은 회색
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        cmap.set_bad("lightgray")

        # anno에서 nan은 공백으로 변경
        df_for_anno = df_for_anno.fillna('')

        # plot
        plt.figure(figsize=(TUPLE_FIGSIZE))
        ax = sns.heatmap(
            df_for_heatmap,
            annot=df_for_anno,
            fmt="",
            cmap=custom_cmap,
            cbar=True,
            yticklabels=False,
            cbar_kws={'shrink': 0.6},
            vmin=0,
            annot_kws={"size": annot_fontsize}
        )

        # T/L 레이블 수동 삽입
        y_pos = np.arange(len(T_vals)) + 0.5
        prev_t = None
        for i, (t, l) in enumerate(zip(T_vals, L_vals)):
            t = f"{t}: {dict_task_label2short_explanation[t]}"
            t_text = t if t != prev_t else ""
            ax.text(-0.3, y_pos[i], t_text, va='center', ha='right', fontsize=10, fontweight='bold')
            ax.text(-0.1, y_pos[i], l, va='center', ha='right', fontsize=10, fontweight='bold')
            prev_t = t

        # x축 레이블 bold
        new_xticklabels = [f"{label}: {short_explanation}" for label, short_explanation in dict_domain_label2short_explanation.items()]
        ax.set_xticklabels(new_xticklabels, rotation=0, ha='center' , fontweight='bold')

        # 구분선
        t_change_idx = [i for i in range(1, len(T_vals)) if T_vals[i] != T_vals[i-1]]
        l_change_idx = [i for i in range(1, len(L_vals)) if L_vals[i] != L_vals[i-1]]
        for y in t_change_idx:
            ax.hlines(y, *ax.get_xlim(), colors='black', linewidth=1)
        for y in l_change_idx:
            if y not in t_change_idx:
                ax.hlines(y, *ax.get_xlim(), colors='gray', linestyles='dotted', linewidth=1)

        # x축 세로선
        n_cols = len(df_for_heatmap.columns)
        for x in range(1, n_cols):
            ax.vlines(x, *ax.get_ylim(), colors='lightgray', linewidth=1)

        # 라벨 및 타이틀
        ax.set_ylabel("")
        ax.set_xlabel("", fontsize=11, fontweight='bold')
        plt.title(title, pad=20, fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)



    @staticmethod
    def TLDCMappingHeatmap(pdf_filtered: pd.DataFrame):
  
        print("pdf",pdf_filtered.columns)

        dict_llm_name2df_records = {}

        for answerer_llm_alias in pdf_filtered['answerer_llm_alias'].unique().tolist():
            dict_llm_name2df_records[answerer_llm_alias] = pdf_filtered[pdf_filtered['answerer_llm_alias'] == answerer_llm_alias]
            dict_llm_name2df_records[answerer_llm_alias].reset_index(drop=True, inplace=True)

        list_task_label = [f"T{i}" for i in range(1, 13)]
        list_domain_label = [f"D{i}" for i in range(1, 11)]
        list_level_label = [f"L{i}" for i in range(1, 4)]
        list_llm_name = list(dict_llm_name2df_records.keys())
  
        dict_llm_name2df = {}    

        print("df_1111 : ")

        for llm_name, df in dict_llm_name2df_records.items():
            list_labels = []
            for t in list_task_label:
                for d in list_domain_label:
                    for l in list_level_label:
                        list_labels.append({
                            "T": t,
                            "D": d,
                            "L": l,
                        })
            df_tld = pd.DataFrame(list_labels)
            df_tld['count'] = 0.0
            df_tld['quality'] = 0.0
            df_tld['cost'] = 0.0
            print("22222 : ", llm_name)
            
            print(len(df))
            
            for index, row in tqdm(df.iterrows()):
                T = row['label_task']
                D = row['label_domain']
                L = row['label_level']
                cost = row['cost']
                quality = row['quality']
                
                normalization_factor = 1 / (len(T) * len(D) * len(L))
                for t_to_count, d_to_count, l_to_count in itertools.product(T, D, L):
                    mask = (
                        (df_tld['T'] == t_to_count) &
                        (df_tld['D'] == d_to_count) &
                        (df_tld['L'] == l_to_count)
                    )
                    df_tld.loc[mask, 'count'] += normalization_factor
                    df_tld.loc[mask, 'cost'] += cost * normalization_factor
                    df_tld.loc[mask, 'quality'] += quality * normalization_factor
            print("33333 : ", llm_name)
            
            dict_llm_name2df[llm_name] = df_tld
            print(f'{llm_name}: {len(dict_llm_name2df[llm_name])}')

           

        #ßdict_llm_name2df['gpt-4.1'].head(2)
        # dict_llm_name2df는 딕셔너리이므로 각 모델별 DataFrame을 개별적으로 저장
        print(f"dict_llm_name2df contains {len(dict_llm_name2df)} models")
        
        # 각 모델별 DataFrame을 개별 CSV 파일로 저장 (선택사항)
        # for model_name, df_model in dict_llm_name2df.items():
        #     filename = f'dict_llm_name2df_{model_name.replace("-", "_")}.csv'
        #     df_model.to_csv(filename, index=False)
        #     print(f"Saved {filename}")



        # 모든 DataFrame을 하나로 합쳐서 저장
        all_data = []
        for model_name, df_model in dict_llm_name2df.items():
            df_with_model = df_model.copy()
            df_with_model['model_name'] = model_name
            all_data.append(df_with_model)
            
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv('all_models_data.csv', index=False)

        ## 위 결과의 key를 llm_alias에서 llm_id로 변경
        dict_llm_id2llm_name = {
        index: llm_name for index, llm_name in enumerate(dict_llm_name2df.keys())
        }
        print(f"{dict_llm_id2llm_name[0]=}")
        dict_llm_id2df = {index: df for index, df in enumerate(dict_llm_name2df.values())}
        dict_llm_id2df[0].head(2)

        ## count가 0이 아닌 것만 모음
        def get_row_from_df_tld(df_tld, T, D, L):
            return df_tld[(df_tld['T'] == T) & (df_tld['D'] == D) & (df_tld['L'] == L)]

        df_all_candidate_tld = pd.DataFrame(columns=['T', 'D', 'L', 'list_dict_list_llm_ids_and_quality_and_cost_efficient', 'count'])

        df_sample_tld = dict_llm_id2df[0]

        for index, current_target_row in tqdm(df_sample_tld[df_sample_tld['count'].ne(0)].iterrows()):
            T = current_target_row['T']
            D = current_target_row['D']
            L = current_target_row['L']
            count = current_target_row['count']

            list_dict_list_llm_id_and_quality_and_cost = []

            for candidate_llm_id, df_tld_to_check in dict_llm_id2df.items():
                row_to_check = get_row_from_df_tld(df_tld_to_check, T, D, L)
                candidate_quality = row_to_check['quality'].values[0]
                candidate_cost = row_to_check['cost'].values[0]
                list_dict_list_llm_id_and_quality_and_cost.append({'list_llm_id': [candidate_llm_id], 'quality': candidate_quality, 'cost': candidate_cost})

            list_dict_list_llm_id_and_quality_and_cost_efficient = filter_efficient_llm(list_dict_list_llm_id_and_quality_and_cost)
            # print(T,D,L)
            # print(list_dict_list_llm_id_and_quality_and_cost_efficient)
            if len(list_dict_list_llm_id_and_quality_and_cost_efficient) == 0:
                print(T,D,L)
                print(list_dict_list_llm_id_and_quality_and_cost_efficient)
                break
     
            df_all_candidate_tld.loc[len(df_all_candidate_tld)] = [T, D, L, list_dict_list_llm_id_and_quality_and_cost_efficient, count]
        df_all_candidate_tld.head(2)
    
        #df_all_candidate_tld.to_dict(orient='records')
        #df_all_candidate_tld['list_dict_list_llm_ids_and_quality_and_cost_efficient'].map(len).value_counts().sort_index()
        
        ## count가 0개가 아닌것만 처음에 모였고,
        ## TLD 조합 내 최적 LLM이 1개인 것들을 별도로 모음
        df_already_efficient_tld = df_all_candidate_tld[df_all_candidate_tld['list_dict_list_llm_ids_and_quality_and_cost_efficient'].map(len) == 1].reset_index(drop=True)
        print("df_already_efficient_tld.shape : ", df_already_efficient_tld.shape)
    

        ## TLD 조합 내 최적 LLM이 1개인 것들의 quality, cost를 누적 (어떠한 조합에서도 누적된 quality, cost는 기본이됨)
        total_quality_of_efficient_tld = 0
        total_cost_of_efficient_tld = 0
        for _, df_already_efficient_tld_row in df_already_efficient_tld.iterrows():
            total_quality_of_efficient_tld += df_already_efficient_tld_row['list_dict_list_llm_ids_and_quality_and_cost_efficient'][0]['quality']
            total_cost_of_efficient_tld += df_already_efficient_tld_row['list_dict_list_llm_ids_and_quality_and_cost_efficient'][0]['cost']
      
        ## TLD 조합 내 최적 LLM이 2개 이상인 것들을 추림
        ## 이것들을 가지고 최적 LLM을 찾아야 함
        df_to_find_efficient_tld = df_all_candidate_tld[df_all_candidate_tld['list_dict_list_llm_ids_and_quality_and_cost_efficient'].map(len) >= 2].reset_index(drop=True)
        print("df_to_find_efficient_tld.shape : ", df_to_find_efficient_tld.shape)

        df_to_find_efficient_tld['list_dict_list_llm_ids_and_quality_and_cost_efficient'].to_list()

        ## TLD 조합 내 최적 LLM이 2개 이상인 것들을 입력해서
        ## 최적을 찾음
        efficient_tld = get_efficient_tld_by_step_by_step(df_to_find_efficient_tld['list_dict_list_llm_ids_and_quality_and_cost_efficient'].to_list())

        ## 2개 이상이었던 efficient_tld에 1개짜리 값 추가
        for dict_efficient_tld in efficient_tld:
            dict_efficient_tld['cost'] += total_cost_of_efficient_tld
            dict_efficient_tld['quality'] += total_quality_of_efficient_tld


        ## 예시로 1개만 만들어 보기 위한 샘플 데이터 확보
        pdf_efficient_tld_sample = efficient_tld[-1]
        pdf_efficient_tld_sample['llm_name'] = [dict_llm_id2llm_name[llm_id] for llm_id in pdf_efficient_tld_sample['list_llm_id']]
        pdf_merged_sample = pd.concat([df_to_find_efficient_tld, pd.DataFrame(pdf_efficient_tld_sample)], axis=1)
        print("pdf+merge", pdf_merged_sample.head(2))

        ## TLD Table 만들기 위해 pivot table 만듬. 이 때의 값은 위 샘플
        pdf_merged_sample_llm_name_as_tld_table = pivot_df_as_tld2tld_as_table(pdf_merged_sample, values='llm_name', aggfunc=lambda x: '\n'.join(x))
        #pdf_merged_sample_llm_name_as_tld_table.to_csv('./processed-data/highest-quality-tldc-all.csv', encoding='utf-8-sig', index=True)
        print(pdf_merged_sample_llm_name_as_tld_table.head(2))
        print("pdf_merged_sample_llm_name_as_tld_tablepdf_merged_sample_llm_n")
        
        
        ## TLD Table 만들기 위해 pivot table 만듬. 이 때의 값은 count이며, heatmap과 같은 색조를 넣기 위해서 사용됨
        df_efficient_tld_num_llm_as_tld_table = pivot_df_as_tld2tld_as_table(pdf_merged_sample, values='count', aggfunc=sum)
        print(df_efficient_tld_num_llm_as_tld_table.head(2))
        print("df_efficient_tld_num_llm_as_tld_tabledf_efficient_tld_num_llm_as_tld_tabledf_efficient_tld_num_llm_as_tld_table")

        ChartView.plot_tld_table(df_efficient_tld_num_llm_as_tld_table, pdf_merged_sample_llm_name_as_tld_table, 'highest_quality')

        return efficient_tld

    

    
    
    def ModelHeatmap(df_filtered: pd.DataFrame):
        st.title(f"모델별 히트맵")
        st.write("모델별 히트맵 입니다.")

        # 설정 초기화
        config = MatrixConfig.create_default()

        df_filtered = df_filtered[df_filtered['answerer_llm_alias'].notna()]

        model_names = df_filtered['answerer_llm_alias'].unique()
        
        model_colors = ["Reds", "Blues", "Greens", "Oranges"]
        for i, model in enumerate(model_names):
            if i < len(model_colors):
                config.model_colors[model] = model_colors[i]
            else:
                config.model_colors[model] = "gray"
        
        # 프로세서와 렌더러 초기화
        processor = MatrixProcessor(config)
        renderer = ChartRenderer(config)
        
        # 모델 선택
        model = st.selectbox("모델 선택", model_names)
        
        # 단일 모델 히트맵 렌더링
        df = processor.create_matrix(df_filtered, model)
        renderer.render_single_model_heatmap(df, model)
        
        # 모델 비교 히트맵 렌더링
        renderer.render_model_comparison(df_filtered, model_names, processor)
        
        # 평가 비중 선택
        weight_type = st.selectbox("평가 비중", ["품질", "가격", "가성비"])
        
        # 최고 모델 히트맵 렌더링
        model_matrices, token_matrices = processor.create_model_matrices(df_filtered, model_names)
        best_values, best_models = processor.calculate_best_values(
            model_matrices, token_matrices, model_names, weight_type
        )
        renderer.render_best_model_heatmap(best_values, best_models, weight_type)

@dataclass
class MatrixConfig:
    """매트릭스 설정을 위한 데이터 클래스"""
    domains: List[str]
    tasks: List[str]
    levels: List[str]
    tasks_with_all_levels: List[str]
    model_colors: Dict[str, str]
    
    @classmethod
    def create_default(cls) -> 'MatrixConfig':
        """기본 설정으로 MatrixConfig 생성"""
        domains = [f'D{i}' for i in range(1, 11)]
        tasks = [f'T{i}' for i in range(1, 12)]
        levels = [f'L{i}' for i in range(1, 3)]
        
        # 각 task별로 L1, L2 세 개의 level을 모두 포함하는 세로축 생성
        tasks_with_all_levels = []
        for task in tasks:
            for level in levels:
                if level == 'L1':
                    tasks_with_all_levels.append(f"{task} {level}")
                tasks_with_all_levels.append(f"{level}")
        
        # 모델별 컬러맵 배열 정의
        model_colors = ["Reds", "Blues", "Greens", "Oranges"]
        
        return cls(domains, tasks, levels, tasks_with_all_levels, {})

class MatrixProcessor:
    """매트릭스 데이터 처리를 담당하는 클래스"""
    
    def __init__(self, config: MatrixConfig):
        self.config = config
    
    def create_matrix(self, data: pd.DataFrame, model: str) -> pd.DataFrame:
        """특정 모델의 매트릭스를 생성합니다."""
        matrix = np.full((len(self.config.tasks_with_all_levels), 
                         len(self.config.domains)), np.nan)
        
        # 해당 모델의 데이터만 필터링
        model_data = data[data['answerer_llm_alias'] == model]
        
        for idx, row in model_data.iterrows():
            d = row.get('label_domain', '')
            t = row.get('label_task', '')
            l = row.get('label_level', '')
            
            try:
                d_idx = self.config.domains.index(d)
                task_level_key = f"{t} {l}"
                t_idx = self.config.tasks_with_all_levels.index(task_level_key)
                score = row['quality']
                
                matrix[t_idx, d_idx] = score
            except ValueError:
                continue
        
        index = pd.Index(self.config.tasks_with_all_levels)
        columns = pd.Index(self.config.domains)
        return pd.DataFrame(matrix, index=index, columns=columns)
    
    def create_model_matrices(self, data: pd.DataFrame, 
                            model_names: List[str]) -> Tuple[Dict[str, np.ndarray], 
                                                           Dict[str, np.ndarray]]:
        """모든 모델의 점수와 토큰 매트릭스를 생성합니다."""
        model_matrices = {m: np.full((len(self.config.tasks_with_all_levels), 
                                    len(self.config.domains)), np.nan) 
                         for m in model_names}
        token_matrices = {m: np.full((len(self.config.tasks_with_all_levels), 
                                    len(self.config.domains)), np.nan) 
                         for m in model_names}
        
        for idx, row in data.iterrows():
            m = row['answerer_llm_alias']
            d = row.get('label_domain', '')
            t = row.get('label_task', '')
            l = row.get('label_level', '')
            tokens = row.get('token', row.get('tokens', 0))
            
            try:
                d_idx = self.config.domains.index(d)
                task_level_key = f"{t} {l}"
                t_idx = self.config.tasks_with_all_levels.index(task_level_key)
                model_matrices[m][t_idx, d_idx] = row['quality']
                token_matrices[m][t_idx, d_idx] = tokens
            except ValueError:
                continue
        
        return model_matrices, token_matrices
    
    def calculate_best_values(self, model_matrices: Dict[str, np.ndarray], 
                            token_matrices: Dict[str, np.ndarray], 
                            model_names: List[str], 
                            weight_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """가중치 타입에 따라 최고 값을 계산합니다."""
        best_values = np.full((len(self.config.tasks_with_all_levels), 
                             len(self.config.domains)), np.nan)
        best_models = np.full((len(self.config.tasks_with_all_levels), 
                             len(self.config.domains)), '', dtype=object)
        
        for i in range(len(self.config.tasks_with_all_levels)):
            for j in range(len(self.config.domains)):
                cell_scores = []
                cell_tokens = []
                
                for m in model_names:
                    score = model_matrices[m][i, j]
                    token = token_matrices[m][i, j]
                    cell_scores.append(score)
                    cell_tokens.append(token)
                
                cell_scores = np.array(cell_scores)
                cell_tokens = np.array(cell_tokens)
                
                if weight_type == "품질":
                    if np.all(np.isnan(cell_scores)):
                        best_values[i, j] = np.nan
                        best_models[i, j] = ''
                    else:
                        idx = np.nanargmax(cell_scores)
                        best_values[i, j] = cell_scores[idx]
                        best_models[i, j] = model_names[idx]
                
                elif weight_type == "가격":
                    valid = (cell_tokens > 0)
                    if not np.any(valid):
                        best_values[i, j] = np.nan
                        best_models[i, j] = ''
                    else:
                        idx = np.argmin(cell_tokens[valid])
                        best_values[i, j] = cell_tokens[valid][idx]
                        best_models[i, j] = np.array(model_names)[valid][idx]
                
                elif weight_type == "가성비":
                    valid = (cell_tokens > 0)
                    ratio = np.zeros_like(cell_scores)
                    ratio[valid] = cell_scores[valid] / cell_tokens[valid] * 100
                    ratio[~valid] = np.nan
                    
                    if np.all(np.isnan(ratio)):
                        best_values[i, j] = np.nan
                        best_models[i, j] = ''
                    else:
                        idx = np.nanargmax(ratio)
                        best_values[i, j] = ratio[idx]
                        best_models[i, j] = model_names[idx]
        
        return best_values, best_models

class ChartRenderer:
    """차트 렌더링을 담당하는 클래스"""
    
    def __init__(self, config: MatrixConfig):
        self.config = config
    
    def render_single_model_heatmap(self, df: pd.DataFrame, model: str) -> None:
        """단일 모델의 히트맵을 렌더링합니다."""
        st.title(f"모델별 평가 결과: {model}")
        st.write("각 셀은 평균 점수(0~1, 소수점 3자리)입니다.")
        
        cmap = self.config.model_colors.get(model, "gray")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            df, 
            annot=True, 
            fmt=".3f", 
            cmap=cmap, 
            linewidths=1.5, 
            linecolor="#888888", 
            ax=ax, 
            vmin=0, 
            vmax=1,
            cbar_kws={"shrink": 0.8, "orientation": "vertical"},
            yticklabels=True,
            xticklabels=True,
            annot_kws={"size": 8}  # 셀 내 텍스트 폰트 사이즈
        )
        
        self._style_heatmap(ax, title_size=14, label_size=10, tick_size=9)
        st.pyplot(fig)
    
    def render_model_comparison(self, data: pd.DataFrame, 
                               model_names: List[str], 
                               processor: MatrixProcessor) -> None:
        """모든 모델의 비교 히트맵을 렌더링합니다."""
        st.title("모델별 평가 결과 비교")
        
        cols = st.columns(len(model_names))
        for idx, model in enumerate(model_names):
            df = processor.create_matrix(data, model)
            
            with cols[idx]:
                st.subheader(model)
                fig, ax = plt.subplots(figsize=(6, 6))
                cmap = self.config.model_colors.get(model, "YlGnBu")
                
                sns.heatmap(
                    df,
                    annot=True,
                    fmt=".3f",
                    cmap=cmap,
                    linewidths=1.5,
                    linecolor="#888888",
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cbar_kws={"shrink": 0.8, "orientation": "vertical"},
                    yticklabels=True,
                    xticklabels=True,
                    annot_kws={"size": 6}  # 작은 차트의 셀 내 텍스트 폰트 사이즈
                )
                
                self._style_heatmap(ax, title_size=10, label_size=8, tick_size=7)
                st.pyplot(fig)
    
    def render_best_model_heatmap(self, best_values: np.ndarray, 
                                best_models: np.ndarray, 
                                weight_type: str) -> None:
        """최고 모델 히트맵을 렌더링합니다."""
        st.title(f"라우팅 품질 Matrix (최고 {weight_type} 모델)")
        
        if weight_type == "품질":
            st.write("각 셀에는 최고 점수를 받은 모델명과 점수가 표시됩니다.")
        elif weight_type == "가격":
            st.write("각 셀에는 비용이 가장 적은 모델명과 토큰 수가 표시됩니다.")
        elif weight_type == "가성비":
            st.write("각 셀에는 가성비(점수/비용)가 가장 높은 모델명과 그 값을 표시합니다.")
        
        # annotation 생성
        annot = np.empty(best_values.shape, dtype=object)
        for i in range(len(self.config.tasks_with_all_levels)):
            for j in range(len(self.config.domains)):
                if best_models[i, j]:
                    annot[i, j] = f"{best_models[i, j]}\n{best_values[i, j]:.3f}"
                else:
                    annot[i, j] = ""
        
        df_best = pd.DataFrame(best_values, 
                             index=pd.Index(self.config.tasks_with_all_levels), 
                             columns=pd.Index(self.config.domains))
        
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            df_best,
            annot=annot,
            fmt="",
            cmap="Reds",
            linewidths=1.5,
            linecolor="#888888",
            ax=ax2,
            vmin=0,
            vmax=1 if weight_type != "가격" else None,
            cbar_kws={"shrink": 0.8, "orientation": "vertical"},
            yticklabels=True,
            xticklabels=True,
            annot_kws={"size": 7}  # 모델명과 점수가 함께 표시되는 셀의 폰트 사이즈
        )
        
        self._style_heatmap(ax2, title_size=14, label_size=10, tick_size=9)
        st.pyplot(fig2)
    
    def _style_heatmap(self, ax: plt.Axes, title_size: int = 12, label_size: int = 10, tick_size: int = 9) -> None:
        """히트맵 스타일을 적용합니다."""
        # 외곽선 추가
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('#888888')
            spine.set_linewidth(2)
        
        # 가로 범주(컬럼명)를 위에 표시
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        # Set tick positions first, then rotate
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # 폰트 사이즈 설정
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        
        # 축 라벨 폰트 사이즈 설정
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=label_size)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=label_size)
        
        # 제목 폰트 사이즈 설정
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=title_size)

    def parse_json_block_safely(self, md_string: str) -> dict:

        if "```json" in md_string:
            # 0. 앞뒤 제거
            md_string = "```json" + md_string.split("```json")[1]
            # 1. 코드 블록 제거
            cleaned = re.sub(r'^```json\s*|\s*```$', '', md_string.strip(), flags=re.IGNORECASE)
        else:
            cleaned = md_string

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


    
