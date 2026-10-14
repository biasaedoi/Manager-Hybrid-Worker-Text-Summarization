import csv
import statistics
import time
from evaluate import compression_ratio, evaluate_summary
from llm_worker import summarize_with_llm_direct  # Pakai worker khusus direct
from data_loader import load_arxiv_sample


def run_direct_experiment(num_docs=10, start_index=0):
  dataset = load_arxiv_sample(start=start_index, n=num_docs)

  results = []

  for doc_id, item in enumerate(dataset):
    article = item['article']

    print('\n===================')
    print('DOCUMENT', doc_id + 1)
    print('===================')

    start = time.time()

    # Langsung lempar seluruh article tanpa chunking & tanpa aggregator
    result = summarize_with_llm_direct(article)

    latency = time.time() - start
    final_summary = result['summary']

    # Evaluation
    metrics = evaluate_summary(
        reference=item['reference'], generated=final_summary
    )

    cr = compression_ratio(article, final_summary)

    # Print Log
    print('\n========== GENERATED SUMMARY ==========\n')
    print(final_summary)

    print('\n========== GROUND TRUTH ==========\n')
    print(item['reference'])

    print('\n========== EVALUATION ==========')
    print(f"ROUGE-1  : {metrics['ROUGE1']:.4f}")
    print(f"ROUGE-L  : {metrics['ROUGEL']:.4f}")
    print(f"BERTScore: {metrics['BERTScore']:.4f}")
    print(f"BGE-M3   : {metrics['BGE_M3']:.4f}")
    print(f'Compression Ratio : {cr:.4f}')

    print('\n========== LATENCY ==========')
    print(f'Total document latency : {latency:.4f} s')
    print(f"  TTFT     : {result['ttft']} s")
    print(f"  TPOT     : {result['tpot']} s")

    results.append({
        'document': doc_id + 1,
        'latency': round(latency, 5),
        'LLM_TTFT': result['ttft'],
        'LLM_TPOT': result['tpot'],
        'ROUGE1': metrics['ROUGE1'],
        'ROUGEL': metrics['ROUGEL'],
        'BERTScore': metrics['BERTScore'],
        'BGE_M3': metrics['BGE_M3'],
        'CompressionRatio': cr,
        'generated_summary': final_summary,
        'reference_summary': item['reference'],
    })

  return results


if __name__ == '__main__':
  results = run_direct_experiment(num_docs=10, start_index=0)

  with open('results_direct.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

  print('\nDONE DIRECT EXPERIMENT')