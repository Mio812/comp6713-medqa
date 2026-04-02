import torch
import warnings
from medqa.data.loader import load_all
from medqa.data.preprocessor import split_dataset
from medqa.evaluation.metrics import evaluate, print_results
from medqa.models.bert_qa import PubMedBERTQA
from medqa.models.baseline import TFIDFBaseline

# 忽略烦人的 HuggingFace 警告
warnings.filterwarnings("ignore")

try:
    from rouge_score import rouge_scorer
    from bert_score import score as calculate_bert_score
except ImportError:
    print("⚠️ 缺少必要库！请先运行: uv add rouge-score bert-score")
    exit(1)

# 初始化 ROUGE 计分器
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def calculate_rouge_l(preds, golds):
    if not preds or not golds: return 0.0
    scores = [scorer.score(target=g, prediction=p)['rougeL'].fmeasure for p, g in zip(preds, golds)]
    return sum(scores) / len(scores)

def get_safe_bertscore(preds, golds):
    """安全的 BERTScore 计算器，绕过 Python 3.11 的超大整数溢出 Bug"""
    if not preds or not golds: return 0.0
    try:
        # 强制使用 roberta-base，避免 deberta-xlarge 触发 int 溢出
        P, R, F1 = calculate_bert_score(preds, golds, model_type="roberta-base", lang="en", verbose=False)
        return F1.mean().item()
    except Exception as e:
        print(f"[BERTScore Error] {e}")
        return 0.0

# 1. 加载数据
print('[Step 1] Loading test data...')
_, test = split_dataset(load_all())
test_subset = test[:100] 
questions = [r['question'] for r in test_subset]
contexts = [r['context'] for r in test_subset]
golds = [r['answer'] for r in test_subset]

# 2. 评估 Baseline
print('\n[Step 2] Evaluating Baseline...')
bl_model = TFIDFBaseline()
res_bl = {'mean_em': 0.0, 'mean_f1': 0.0, 'bertscore': 0.0, 'rouge_l': 0.0}
if bl_model.load():
    bl_preds = [bl_model.predict(q)['predicted_answer'] for q in questions]
    # 关闭底层会报错的 bertscore
    res_bl = evaluate(bl_preds, golds, use_bertscore=False)
    # 使用我们自己的安全版
    res_bl['bertscore'] = get_safe_bertscore(bl_preds, golds) 
    res_bl['rouge_l'] = calculate_rouge_l(bl_preds, golds)
    print_results(res_bl, model_name='Baseline')

# 3. 评估 BERT
print('\n[Step 3] Evaluating Fine-tuned BERT...')
bert_model = PubMedBERTQA()
res_bert = {'mean_em': 0.0, 'mean_f1': 0.0, 'bertscore': 0.0, 'rouge_l': 0.0}
if bert_model.load():
    bert_preds_data = bert_model.batch_predict(questions, contexts)
    bert_preds = [p['predicted_answer'] for p in bert_preds_data]
    res_bert = evaluate(bert_preds, golds, use_bertscore=False)
    res_bert['bertscore'] = get_safe_bertscore(bert_preds, golds)
    res_bert['rouge_l'] = calculate_rouge_l(bert_preds, golds)
    print_results(res_bert, model_name='Fine-tuned BERT')

# 4. 评估 RAG
print('\n[Step 4] Evaluating RAG/LLM (Qwen-14B)...')
res_rag = {'mean_em': 0.0, 'mean_f1': 0.0, 'bertscore': 0.0, 'rouge_l': 0.0}
try:
    from medqa.models.llm_qa import LocalLLM
    rag_model = LocalLLM()
    rag_preds = []
    
    print("  -> Loading Qwen-14B weights to GPU...")
    rag_model.load()
    
    print(f"  -> Predicting 100 samples with Qwen-14B, this might take a few minutes...")
    for i, (q, c) in enumerate(zip(questions, contexts)):
        res = rag_model.predict(question=q, context=c)
        ans = res.get('predicted_answer', res.get('answer', ''))
        rag_preds.append(ans)
        if (i + 1) % 10 == 0:
            print(f"     ... processed {i + 1}/100")
            
    res_rag = evaluate(rag_preds, golds, use_bertscore=False)
    res_rag['bertscore'] = get_safe_bertscore(rag_preds, golds)
    res_rag['rouge_l'] = calculate_rouge_l(rag_preds, golds)
    print_results(res_rag, model_name='RAG (Qwen-14B)')

except Exception as e:
    print(f'⚠️ Skipping RAG due to error: {e}')

# 5. 最终汇总打印
print('\n' + '='*38 + ' FINAL COMPARISON (n=100) ' + '='*38)
print(f'Model          | EM      | F1      | BERTScore | ROUGE-L')
print(f'--------------------------------------------------------------------------------')
print(f'Baseline       | {res_bl.get("mean_em", 0):.4f}  | {res_bl.get("mean_f1", 0):.4f}  | {res_bl.get("bertscore", 0):.4f}    | {res_bl.get("rouge_l", 0):.4f}')
print(f'BERT (Finetune)| {res_bert.get("mean_em", 0):.4f}  | {res_bert.get("mean_f1", 0):.4f}  | {res_bert.get("bertscore", 0):.4f}    | {res_bert.get("rouge_l", 0):.4f}')
print(f'RAG (Qwen-14B) | {res_rag.get("mean_em", 0):.4f}  | {res_rag.get("mean_f1", 0):.4f}  | {res_rag.get("bertscore", 0):.4f}    | {res_rag.get("rouge_l", 0):.4f}')
print('='*100)

import json

# 整理你要保存的数据字典
final_results_dict = {
    "Baseline": {
        "EM": res_bl.get("mean_em", 0),
        "F1": res_bl.get("mean_f1", 0),
        "BERTScore": res_bl.get("bertscore", 0),
        "ROUGE-L": res_bl.get("rouge_l", 0)
    },
    "Fine-tuned BERT": {
        "EM": res_bert.get("mean_em", 0),
        "F1": res_bert.get("mean_f1", 0),
        "BERTScore": res_bert.get("bertscore", 0),
        "ROUGE-L": res_bert.get("rouge_l", 0)
    },
    "RAG (Qwen-14B)": {
        "EM": res_rag.get("mean_em", 0),
        "F1": res_rag.get("mean_f1", 0),
        "BERTScore": res_rag.get("bertscore", 0),
        "ROUGE-L": res_rag.get("rouge_l", 0)
    }
}

# 保存为 JSON 文件
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(final_results_dict, f, indent=4)

print("✅ 所有评测数据已成功保存至 evaluation_results.json")