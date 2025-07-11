from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()

import os
import time
import csv
from typing import List, Dict, Any
from llama_index.core import (
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter

# ---------- Configuration ---------- #

# Load document once
print("Loading document...")
documents = SimpleDirectoryReader(input_files=["NPC07May2025.pdf"]).load_data()


# ---------- Define Single Experiment Runner ---------- #
def run_experiment(
    llm_model: str,
    embed_model: str,
    chunk_size: int,
    parser_class,
    parser_name: str,
    selector_class,
    selector_name: str,
    queries: List[str],
    results_file="router_engine_experiments.csv"
):
    print(f"\n Running experiment:")
    print(f"  - LLM: {llm_model}")
    print(f"  - Embedding: {embed_model}")
    print(f"  - Chunk Size: {chunk_size}")
    print(f"  - Parser: {parser_name}")
    print(f"  - Selector: {selector_name}")

    # Set models
    Settings.llm = OpenAI(model=llm_model)
    Settings.embed_model = OpenAIEmbedding(model=embed_model)

    # Create nodes
    splitter = parser_class(chunk_size=chunk_size)
    nodes = splitter.get_nodes_from_documents(documents)

    # Indexes
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
    vector_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(query_engine=summary_engine, description="Summarization tool")
    vector_tool = QueryEngineTool.from_defaults(query_engine=vector_engine, description="Context retrieval tool")

    router = RouterQueryEngine(
        selector=selector_class.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True,
    )

    # Run queries
    results: List[Dict[str, Any]] = []
    for query in queries:
        print(f"\nQuery: {query}")
        start_time = time.time()
        response = router.query(query)
        duration = time.time() - start_time

        results.append({
            "llm_model": llm_model,
            "embedding_model": embed_model,
            "chunk_size": chunk_size,
            "parser": parser_name,
            "selector": selector_name,
            "query": query,
            "response": str(response),
            "source_nodes": len(response.source_nodes),
            "time_taken_sec": round(duration, 2),
        })

    # Save (append) results
    write_mode = 'a' if os.path.exists(results_file) else 'w'
    with open(results_file, write_mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_mode == 'w':
            writer.writeheader()
        writer.writerows(results)

    print(f"Results appended to {results_file}\n")


# ---------- Example Usage: One Config at a Time ---------- #
QUERIES = [
    "What is the summary of the document?",
    "How did the Indian government coordinate among the Army, Air Force, and Navy during the operation?",
    "What are the latest updates on India’s Gaganyaan mission and other defence-related technology initiatives?",
    "What were the international reactions to India’s actions?"
]

# Call this for each separate run manually
run_experiment(
    llm_model="gpt-3.5-turbo",
    embed_model="text-embedding-ada-002",
    chunk_size=512,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-3.5-turbo",
    embed_model="text-embedding-ada-002",
    chunk_size=1024,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-3.5-turbo",
    embed_model="text-embedding-ada-002",
    chunk_size=2048,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-3.5-turbo",
    embed_model="text-embedding-3-small",
    chunk_size=512,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-3.5-turbo",
    embed_model="text-embedding-3-small",
    chunk_size=1024,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-3.5-turbo",
    embed_model="text-embedding-3-small",
    chunk_size=2048,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4",
    embed_model="text-embedding-ada-002",
    chunk_size=512,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4",
    embed_model="text-embedding-ada-002",
    chunk_size=1024,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4",
    embed_model="text-embedding-ada-002",
    chunk_size=2048,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4",
    embed_model="text-embedding-3-small",
    chunk_size=512,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)


run_experiment(
    llm_model="gpt-4",
    embed_model="text-embedding-3-small",
    chunk_size=1024,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4",
    embed_model="text-embedding-3-small",
    chunk_size=2048,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4-turbo",
    embed_model="text-embedding-ada-002",
    chunk_size=512,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4-turbo",
    embed_model="text-embedding-ada-002",
    chunk_size=1024,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4-turbo",
    embed_model="text-embedding-ada-002",
    chunk_size=2048,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4-turbo",
    embed_model="text-embedding-3-small",
    chunk_size=512,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4-turbo",
    embed_model="text-embedding-3-small",
    chunk_size=1024,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

run_experiment(
    llm_model="gpt-4-turbo",
    embed_model="text-embedding-3-small",
    chunk_size=2048,
    parser_class=SentenceSplitter,
    parser_name="sentence",
    selector_class=LLMSingleSelector,
    selector_name="llm",
    queries=QUERIES,
)

# Calculate Cohen's d for effect size
def cohen_d(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    return mean_diff / pooled_std if pooled_std != 0 else np.nan

# Group data by LLM model
llm_models = df["llm_model"].unique()
llm_groups = {model: df[df["llm_model"] == model]["time_taken_sec"] for model in llm_models}

# Check normality for each group
print("Normality Check (Shapiro-Wilk Test):")
for model, group in llm_groups.items():
    stat, p_shapiro = shapiro(group)
    print(f"{model}: stat={stat:.2f}, p={p_shapiro:.3f}")
    if p_shapiro < 0.05:
        print(f"Warning: {model} time_taken_sec may not be normally distributed.")

# Perform pairwise t-tests with Welch's correction and Bonferroni adjustment
print("\nPairwise T-tests for LLM Model Time Differences (Welch's t-test):")
results = []
alpha = 0.05
bonferroni_alpha = alpha / len(list(combinations(llm_models, 2)))  # Adjust for multiple tests
for model1, model2 in combinations(llm_models, 2):
    # Check variances
    stat, p_levene = levene(llm_groups[model1], llm_groups[model2])
    equal_var = p_levene >= 0.05
    
    # Perform t-test
    t_stat, p_value = ttest_ind(llm_groups[model1], llm_groups[model2], equal_var=equal_var)
    d = cohen_d(llm_groups[model1], llm_groups[model2])
    mean_diff = np.mean(llm_groups[model1]) - np.mean(llm_groups[model2])
    
    results.append({
        "Model 1": model1,
        "Model 2": model2,
        "Mean Diff (s)": mean_diff,
        "T-Statistic": t_stat,
        "P-Value": p_value,
        "Cohen's d": d,
        "Significant": p_value < bonferroni_alpha
    })
    print(f"{model1} vs {model2}: t={t_stat:.2f}, p={p_value:.3f}, Cohen's d={d:.2f}, "
          f"Mean Diff={mean_diff:.2f}s, Significant={p_value < bonferroni_alpha}")

# Create DataFrame for results
results_df = pd.DataFrame(results)

# Bar Plot with T-Test Annotations
avg_time = df.groupby("llm_model")["time_taken_sec"].mean().reset_index()
fig = px.bar(
    avg_time,
    x="llm_model",
    y="time_taken_sec",
    title="Average Query Time for Defense Reports by LLM Model",
    labels={"llm_model": "LLM Model", "time_taken_sec": "Avg Time (seconds)"},
    color="llm_model",
    color_discrete_sequence=px.colors.qualitative.Set2,
    text_auto=".2f"
)

# Add t-test annotations
annotations = []
for idx, row in results_df.iterrows():
    if row["Significant"]:
        annotations.append(
            dict(
                x=0.5,
                y=-0.15 - idx * 0.05,
                xref="paper",
                yref="paper",
                text=f"{row['Model 1']} vs {row['Model 2']}: t={row['T-Statistic']:.2f}, p={row['P-Value']:.3f}",
                showarrow=False,
                font=dict(size=10)
            )
        )
fig.update_layout(
    showlegend=False,
    template="plotly_white",
    title_x=0.5,
    font=dict(size=12),
    annotations=annotations
)
fig.write_html("avg_time_by_llm.html")
