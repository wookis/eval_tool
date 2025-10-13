import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(layout="centered")

# 파일 경로
dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
result_files = [f for f in os.listdir(dataset_dir) if f.startswith('eval_results_') and f.endswith('.json')]

if not result_files:
    st.error('평가 결과 파일이 없습니다.')
    st.stop()

result_file = st.selectbox('결과 파일 선택', result_files)
RESULT_PATH = os.path.join(dataset_dir, result_file)

# 데이터 로드
with open(RESULT_PATH, encoding='utf-8') as f:
    results = json.load(f)

# D1~D12, T1~T11 정의
domains = [f'D{i}' for i in range(1, 11)]
tasks = [f'T{i}' for i in range(1, 12)]
levels = ['1', '2', '3']

# 모델 선택
model_names = sorted(list(set(r['model'] for r in results)))
model = st.selectbox("모델 선택", model_names)

# matrix 생성
matrix = np.full((len(tasks), len(domains)), np.nan)

for r in results:
    if r['model'] != model:
        continue
    d = r.get('domain', '')
    t = r.get('task', '')
    try:
        d_idx = domains.index(d)
        t_idx = tasks.index(t)
        matrix[t_idx, d_idx] = r['score']
    except ValueError:
        continue

# index, columns를 pd.Index로 명시적으로 변환
index = pd.Index(tasks)
columns = pd.Index(domains)
df = pd.DataFrame(matrix, index=index, columns=columns)

# 모델별 컬러맵 지정
model_cmap = {
    "gpt-4o": "Oranges",
    "midm2.x": "Blues",
    "GPT K": "Greens"
}
#cmap = model_cmap.get(model, "YlGnBu")
cmap = model_cmap.get(model, "gray")

st.title(f"모델별 평가 결과: {model}")
st.write("각 셀은 평균 점수(0~1, 소수점 3자리)입니다.")

fig, ax = plt.subplots(figsize=(14, 7))
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
    xticklabels=True
)
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
st.pyplot(fig) 
#####################################################

st.title("모델별 평가 결과 비교")

cols = st.columns(len(model_names))
for idx, model in enumerate(model_names):
    # matrix 생성
    matrix = np.full((len(tasks), len(domains)), np.nan)
    for r in results:
        if r['model'] != model:
            continue
        d = r.get('domain', '')
        t = r.get('task', '')
        try:
            d_idx = domains.index(d)
            t_idx = tasks.index(t)
            matrix[t_idx, d_idx] = r['score']
        except ValueError:
            continue
    index = pd.Index(tasks)
    columns = pd.Index(domains)
    df = pd.DataFrame(matrix, index=index, columns=columns)

    with cols[idx]:
        st.subheader(model)
        fig, ax = plt.subplots(figsize=(6, 4))
        cmap = model_cmap.get(model, "YlGnBu")
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
            xticklabels=True
        )
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('#888888')
            spine.set_linewidth(2)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        # Set tick positions first, then rotate
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)



###################################################################
# 모델 통합 노출 히트맵
# 모델별로 가장 높은 score와 해당 모델명을 표시하는 히트맵 생성
# 평가 비중 선택
weight_type = st.selectbox("평가 비중", ["품질", "가격", "가성비"])

# 1. 모델별로 matrix와 token matrix를 만듦
model_matrices = {m: np.full((len(tasks), len(domains)), np.nan) for m in model_cmap.keys()}
token_matrices = {m: np.full((len(tasks), len(domains)), np.nan) for m in model_cmap.keys()}

for r in results:
    m = r['model']
    d = r.get('domain', '')
    t = r.get('task', '')
    tokens = r.get('token', r.get('tokens', 0))
    try:
        d_idx = domains.index(d)
        t_idx = tasks.index(t)
        model_matrices[m][t_idx, d_idx] = r['score']
        token_matrices[m][t_idx, d_idx] = tokens
    except ValueError:
        continue

best_values = np.full((len(tasks), len(domains)), np.nan)
best_models = np.full((len(tasks), len(domains)), '', dtype=object)

for i in range(len(tasks)):
    for j in range(len(domains)):
        cell_scores = []
        cell_tokens = []
        for m in model_cmap.keys():
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
                best_models[i, j] = list(model_cmap.keys())[idx]
        elif weight_type == "가격":
            valid = (cell_tokens > 0)
            if not np.any(valid):
                best_values[i, j] = np.nan
                best_models[i, j] = ''
            else:
                idx = np.argmin(cell_tokens[valid])
                best_values[i, j] = cell_tokens[valid][idx]
                best_models[i, j] = np.array(list(model_cmap.keys()))[valid][idx]
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
                best_models[i, j] = list(model_cmap.keys())[idx]

# annotation 생성
annot = np.empty(best_values.shape, dtype=object)
for i in range(len(tasks)):
    for j in range(len(domains)):
        if best_models[i, j]:
            annot[i, j] = f"{best_models[i, j]}\n{best_values[i, j]:.3f}"
        else:
            annot[i, j] = ""

df_best = pd.DataFrame(best_values, index=pd.Index(tasks), columns=pd.Index(domains))

st.title(f"라우팅 품질 Matrix (최고 {weight_type} 모델)")
if weight_type == "품질":
    st.write("각 셀에는 최고 점수를 받은 모델명과 점수가 표시됩니다.")
elif weight_type == "가격":
    st.write("각 셀에는 비용이 가장 적은 모델명과 토큰 수가 표시됩니다.")
elif weight_type == "가성비":
    st.write("각 셀에는 가성비(점수/비용)가 가장 높은 모델명과 그 값을 표시합니다.")

fig2, ax2 = plt.subplots(figsize=(14, 7))
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
    xticklabels=True
)
for _, spine in ax2.spines.items():
    spine.set_visible(True)
    spine.set_color('#888888')
    spine.set_linewidth(2)
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
# Set tick positions first, then rotate
ax2.set_xticks(ax2.get_xticks())
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
st.pyplot(fig2)
