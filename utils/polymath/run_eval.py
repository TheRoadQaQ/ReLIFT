import os
import re
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from .scripts import math_equal
from langdetect import detect_langs

language_list = ["en", "zh", "ar", "bn", "de", "es", "fr", "id", "it", "ja", "ko", "ms", "pt", "ru", "sw", "te", "th", "vi", ]
level_list = ["low", "middle", "high", "top"]


def initial_score_json(score_file):
    initial_data = {}
    for _lang in language_list:
        initial_data[_lang] = {}
        for _level in level_list:
            initial_data[_lang][_level] = \
            {
                "accuracy": None,
                "thinking_lang_cons": None,
                "answer_lang_cons": None,
            }
        initial_data[_lang]["benchmark_weighted_acc"] = None
        initial_data[_lang]["benchmark_thinking_lang_cons"] = None
        initial_data[_lang]["benchmark_answer_lang_cons"] = None
    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, ensure_ascii=False, indent=4)


def extract_boxed_content(text):
    pattern = re.compile(r'boxed{')
    text = text.replace(' ', '')

    matches = pattern.finditer(text)
    results = []
    for match in matches:
        start_pos = match.end()
        brace_count = 1
        i = start_pos
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            results.append(text[start_pos:i-1])
    return results


def evaluation(args):
    model = args.model
    language = args.language
    level = args.level
    print(model, language, level)

    output_file = f"../output/{model}/{level}/{language}.jsonl"
    try:
        data = []
        with open(output_file, 'r', encoding='utf-8') as file:
            for line in file:
                item = json.loads(line)
                data.append(item)
    except FileNotFoundError:
        print(f"The file '{output_file}' does not exist.")

    if len(data) < 125:
        print(f"Warning! Test data is incomplete, current data size: {len(data)}")
    elif len(data) > 125:
        print(f"Warning! Test data is redundant, current data size: {len(data)}")
    else:
        pass


    acc, thinking_lang_cons, answer_lang_cons = 0, 0, 0
    for i in tqdm(range(len(data))):
        idx = data[i]["idx"]
        question = data[i]["question"]
        answer = data[i]["answer"]
        thinking_pred = data[i]["thinking_pred"]
        answer_pred = data[i]["answer_pred"]

        ### answer extraction & correctness judgement
        extracted_pred = extract_boxed_content(answer_pred)
        extracted_pred = extracted_pred[0] if len(extracted_pred) > 0 else None
        acc_binary = math_equal(extracted_pred, answer)
        acc += 1 if acc_binary else 0
        
        ### language consistency judgement
        re_thinking_pred = re.sub(r'\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]', '', thinking_pred, flags=re.DOTALL)
        re_thinking_pred = re.sub(r'\$[^$]*\$', '', re_thinking_pred)
        if len(re_thinking_pred) <= 15:
            thinking_lang_cons_binary = True
        else:
            thinking_lang_cons_binary = False
            try:
                lang_prob = detect_langs(re_thinking_pred)
                detect_lang = "zh-cn" if language == "zh" else language
                thinking_lang = [lang.lang for lang in lang_prob]
                thinking_lang_cons_binary = True if (len(lang_prob) == 1 and detect_lang in thinking_lang) else False
            except:
                pass

        re_answer_pred = re.sub(r'\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]', '', answer_pred, flags=re.DOTALL)
        re_answer_pred = re.sub(r'\$[^$]*\$', '', re_answer_pred)
        if len(re_answer_pred) <= 15:
            answer_lang_cons_binary = True
        else:
            answer_lang_cons_binary = False
            try:
                lang_prob = detect_langs(re_answer_pred)
                detect_lang = "zh-cn" if language == "zh" else language
                answer_lang = [lang.lang for lang in lang_prob]
                answer_lang_cons_binary = True if (len(lang_prob) == 1 and detect_lang in answer_lang) else False
            except:
                pass
        
        thinking_lang_cons += 1 if thinking_lang_cons_binary else 0
        answer_lang_cons += 1 if answer_lang_cons_binary else 0
    
    acc = round(acc / len(data) * 100, 1)
    thinking_lang_cons = round(thinking_lang_cons / len(data) * 100, 1)
    answer_lang_cons = round(answer_lang_cons / len(data) * 100, 1)


    print(f"Test Data Size: {len(data)}\n"
        f"Accuracy (%) = {acc}\n"
        f"Language Consistency (thinking) (%) = {thinking_lang_cons}\n"
        f"Language Consistency (answer) (%) = {answer_lang_cons}")
    print("*"*30)


    ### save results
    score_file = os.path.join(f"../output/{model}", "score.json")
    if not os.path.isfile(score_file):
        initial_score_json(score_file)
    with open(score_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data[language][level]["accuracy"] = acc
        data[language][level]["thinking_lang_cons"] = thinking_lang_cons
        data[language][level]["answer_lang_cons"] = answer_lang_cons
    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    ### update benchmark score
    with open(score_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        try:
            benchmark_weighted_acc = sum([(2 ** i) * data[language][_level]["accuracy"] for i, _level in enumerate(level_list)]) / 15
            data[language]["benchmark_weighted_acc"] = round(benchmark_weighted_acc, 1)
            benchmark_thinking_lang_cons = sum([data[language][_level]["thinking_lang_cons"] for i, _level in enumerate(level_list)]) / 4
            data[language]["benchmark_thinking_lang_cons"] = round(benchmark_thinking_lang_cons, 1)
            benchmark_answer_lang_cons = sum([data[language][_level]["answer_lang_cons"] for i, _level in enumerate(level_list)]) / 4
            data[language]["benchmark_answer_lang_cons"] = round(benchmark_answer_lang_cons, 1)
            print("+-------------------------------+")
            print(f"Update Benchmark Score {model} - {language}")
            print(f"Benchmark Weighted Accuracy: {round(benchmark_weighted_acc, 1)}")
            print(f"Benchmark Thinking Language Consistency: {round(benchmark_thinking_lang_cons, 1)}")
            print(f"Benchmark Answer Language Consistency: {round(benchmark_answer_lang_cons, 1)}")
            print("+-------------------------------+")
        except:
            pass
    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--level', type=str, required=True)

    args = parser.parse_args()
    evaluation(args)