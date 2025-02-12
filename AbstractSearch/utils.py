
def reciprocal_rank_fusion(bm25_results, faiss_results):
    result_set = set([i[0] for i in bm25_results] + [i[0] for i in faiss_results])
    rank1 = {result[0]: rank for rank, result in enumerate(bm25_results, 1)}
    rank2 = {result[0]: rank for rank, result in enumerate(faiss_results, 1)}
    combined_scores = {}
    for result in result_set:
        rr1 = 1 / rank1.get(result) if rank1.get(result) else 0
        rr2 = 1 / rank2.get(result) if rank2.get(result) else 0
        combined_scores[result] = rr1 + rr2
    ranked_results = sorted(combined_scores, key=combined_scores.get, reverse=True)
    return ranked_results
