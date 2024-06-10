import pickle
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from vllm import LLM, SamplingParams
import re
import random

async def llama_callable(engine, string_input, system_prompt=None, temperature=0.8, max_tokens=300, generation_args={}):
    """
    Call the Llama model with the given input and parameters.

    Args:
        engine (AsyncLLMEngine): The LLM engine to use.
        string_input (str): The input string to the model.
        system_prompt (str, optional): The system prompt to prepend to the input.
        temperature (float, optional): Sampling temperature.
        max_tokens (int, optional): Maximum number of tokens to generate.
        generation_args (dict, optional): Additional generation arguments.

    Returns:
        str: The generated response.
    """
    string_formatted = "<|begin_of_text|>" 
    if system_prompt:
        string_formatted += f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"
    string_formatted += f"<|start_header_id|>user<|end_header_id|>{string_input}<|eot_id|>"
    string_formatted += f"<|start_header_id|>assistant<|end_header_id|>"

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, **generation_args, stop_token_ids = [128000, 128001], stop = ["<|eot_id|>"])
    request_id = str(random.randint(0, 10000000) + 10000000)
    results_generator = engine.generate(string_formatted, sampling_params, request_id)
    result = ""
    async for request_output in results_generator:
        result = request_output.outputs[0].text
    return result

def extract_items(messy_list):
    """
    Extract items from a string containing a list of options.

    Args:
        messy_list (str): The string containing the list of options.

    Returns:
        list: A list of extracted items.
    """
    pattern = r'(?<!\\)"(.*?)(?<!\\)"|\'(.*?)(?<!\\)\''
    matches = re.findall(pattern, messy_list)
    return [match[0] if match[0] else match[1] for match in matches]

def extract_model_performance(models, partitions, data, model_perf_base_path_template, category="question_format"):
    total_count = {cat: 0 for cat in set(cat for partition in partitions for cat in data[partition][category].unique())}
    performance = {model: {cat: 0 for cat in total_count} for model in models}

    # Calculate total counts for each category across all data partitions
    for partition in partitions:
        for cat in data[partition][category].unique():
            total_count[cat] += len(data[partition][data[partition][category] == cat])
    
    # Calculate correct counts for each model
    for model in models:
        correct_count = {cat: 0 for cat in total_count}
        for partition in partitions:
            model_perf_path = model_perf_base_path_template.replace("<PART>", partition).replace("{model}", model)
            with open(model_perf_path, "rb") as f:
                model_perf = pickle.load(f)["accuracy"].values
            correct_indices = model_perf == 1
            filtered_data = data[partition].iloc[correct_indices]
            
            for cat in filtered_data[category].unique():
                correct_count[cat] += len(filtered_data[filtered_data[category] == cat])
        
        # Normalize by total counts to compute accuracy
        for cat in correct_count:
            if total_count[cat] > 0:
                performance[model][cat] = correct_count[cat] / total_count[cat]
    
    return performance

def plot_model_mistakes(performance, models, title):
    fig, ax = plt.subplots(figsize=(16, 6))  # Increase figure size

    # Calculate mean mistake for each model
    model_mean_mistakes = {
        model: np.mean([1 - performance[model][cat] for cat in performance[model].keys()])
        for model in models
    }
    # Make gpt models appear first by setting mean to 0
    model_mean_mistakes["gpt"] = 0
    
    # Sort models from best to worst performing
    models = sorted(models, key=lambda model: model_mean_mistakes[model])
    
    x = np.arange(len(models))
    width = 0.7 / len(performance[models[0]])  # Adjust the width dynamically
    
    color_map = plt.get_cmap('tab10')
    
    # Ensure consistent colors across categories
    all_categories = sorted(set(cat for model in models for cat in performance[model]))
    category_colors = {cat: color_map(i % 10) for i, cat in enumerate(all_categories)}
    
    # Plot each model individually
    for i, model in enumerate(models):
        sorted_categories = sorted(performance[model].keys(), key=lambda cat: 1 - performance[model][cat])
        model_mistakes = [1 - performance[model][cat] for cat in sorted_categories]
        
        for j, cat in enumerate(sorted_categories):
            ax.bar(x[i] + width * j, model_mistakes[j], width, label=cat if i == 0 else "", color=category_colors[cat])

    ax.set_xticks(x + width * (len(performance[models[0]]) / 2))
    model_formatted_names = [model.replace("_", " ") for model in models]
    model_formatted_names = [model.replace("gpt", "GPT-4V") for model in model_formatted_names]
    # wrap model names if they are too long
    wrapped_model_names = [textwrap.fill(model, width=16) for model in model_formatted_names]
    ax.set_xticklabels(wrapped_model_names, ha="center", fontsize=20)
    ax.set_ylabel("Mistake Rate", fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    # Adding black border to the chart (not edge of image)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Adjust legend inside the graph
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(all_categories)], labels[:len(all_categories)], title=title, loc='upper left', fontsize=18, title_fontsize='18', ncols=2)
    # make legend background fully transparent
    ax.get_legend().get_frame().set_alpha(0)
    plt.tight_layout()
    plt.show()
