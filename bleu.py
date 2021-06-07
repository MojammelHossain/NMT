import sacrebleu

def bleu_score(tars_sentences, pred_sentences):
  bleu = sacrebleu.corpus_bleu(tars_sentences, [pred_sentences])
  return bleu.score