from modules.IR_module import IR_module

ir_module = IR_module()
query = "상고심 계류중에 사망한 영생교 교주의 사망원인은 무엇인가?"
top_documents, top_scores = ir_module.search(query, topk=10)
print("AFTER Sentence-BERT")
ir_module.reranking(query,top_documents,top_scores, top_k=10)