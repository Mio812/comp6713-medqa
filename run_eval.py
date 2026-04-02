import torch
from medqa.data.loader import load_all
from medqa.data.preprocessor import split_dataset
from medqa.evaluation.metrics import evaluate, print_results
from medqa.models.bert_qa import PubMedBERTQA
from medqa.models.baseline import TFIDFBaseline

# 尝试导入 ROUGE 计算工具
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("⚠️ 缺少 rouge-score 库！请先运行: uv pip install rouge-score")
    exit(1)

# 初始化 ROUGE 计分器
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def calculate_rouge_l(preds, golds):
    """辅助函数：计算一批预测结果的平均 ROUGE-L F1 分数"""
    if not preds or not golds:
        return 0.0
    scores = []
    for p, g in zip(preds, golds):
        # 注意：rouge-score 库的输入顺序通常是 (reference, prediction)
        score = scorer.score(target=g, prediction=p)
        scores.append(score['rougeL'].fmeasure)
    return sum(scores) / len(scores)

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
    res_bl = evaluate(bl_preds, golds, use_bertscore=True)
    res_bl['rouge_l'] = calculate_rouge_l(bl_preds, golds) # 计算 ROUGE-L
    print_results(res_bl, model_name='Baseline')

# 3. 评估 BERT
print('\n[Step 3] Evaluating Fine-tuned BERT...')
bert_model = PubMedBERTQA()
res_bert = {'mean_em': 0.0, 'mean_f1': 0.0, 'bertscore': 0.0, 'rouge_l': 0.0}
if bert_model.load():
    bert_preds_data = bert_model.batch_predict(questions, contexts)
    bert_preds = [p['predicted_answer'] for p in bert_preds_data]
    res_bert = evaluate(bert_preds, golds, use_bertscore=True)
    res_bert['rouge_l'] = calculate_rouge_l(bert_preds, golds) # 计算 ROUGE-L
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
        
        # 兼容不同的字典键名
        ans = res.get('predicted_answer', res.get('answer', ''))
        rag_preds.append(ans)
        
        # 每跑 10 条打印一次进度
        if (i + 1) % 10 == 0:
            print(f"     ... processed {i + 1}/100")
            
    res_rag = evaluate(rag_preds, golds, use_bertscore=True)
    res_rag['rouge_l'] = calculate_rouge_l(rag_preds, golds) # 计算 ROUGE-L
    print_results(res_rag, model_name='RAG (Qwen-14B)')

except Exception as e:
    print(f'⚠️ Skipping RAG due to error: {e}')

# 5. 最终汇总打印
print('\n' + '='*38 + ' FINAL COMPARISON (n=100) ' + '='*38)
print(f'Model          | EM      | F1      | BERTScore | ROUGE-L')
print(f'--------------------------------------------------------------------------------')
# 注意：你的 evaluate 返回的 key 可能是 'mean_bertscore' 或 'bertscore'，我这里用了安全的 fallback 获取机制
bl_bs = res_bl.get("mean_bertscore", res_bl.get("bertscore", 0))
bert_bs = res_bert.get("mean_bertscore", res_bert.get("bertscore", 0))
rag_bs = res_rag.get("mean_bertscore", res_rag.get("bertscore", 0))

print(f'Baseline       | {res_bl.get("mean_em", 0):.4f}  | {res_bl.get("mean_f1", 0):.4f}  | {bl_bs:.4f}    | {res_bl.get("rouge_l", 0):.4f}')
print(f'BERT (Finetune)| {res_bert.get("mean_em", 0):.4f}  | {res_bert.get("mean_f1", 0):.4f}  | {bert_bs:.4f}    | {res_bert.get("rouge_l", 0):.4f}')
print(f'RAG (Qwen-14B) | {res_rag.get("mean_em", 0):.4f}  | {res_rag.get("mean_f1", 0):.4f}  | {rag_bs:.4f}    | {res_rag.get("rouge_l", 0):.4f}')
print('='*100)