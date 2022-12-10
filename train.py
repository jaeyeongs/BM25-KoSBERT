from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
from ko_sentence_transformers.models import KoBertTransformer
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model_name = "skt_kobert_model_"

train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
word_embedding_model = KoBertTransformer('monologg/kobert', max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

logging.info("Read STSbenchmark train KorQuADDataset")

train_samples = []
dev_samples = []
test_samples = []

with open('./MergeDataset/korquad_korpair_test.csv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        try:
            # s1, s2, score = line.split('\t')
            s1, s2, score = line.split(',')
            score = score.strip()
            score = float(score) / 5.0
            dev_samples.append(InputExample(texts= [s1,s2], label=score))
        except:
            continue

with open('./MergeDataset/korquad_korpair_test.csv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        try:
            # s1, s2, score = line.split('\t')
            s1, s2, score = line.split(',')
            score = score.strip()
            score = float(score) / 5.0
            test_samples.append(InputExample(texts=[s1, s2], label=score))
        except:
            continue

with open('/workspace/MergeDataset/korquad_korpair_train.csv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        try:
            # s1, s2, score = line.split('\t')
            s1, s2, score = line.split(',')
            score = score.strip()
            score = float(score) / 5.0
            train_samples.append(InputExample(texts=[s1, s2], label=score))
        except:
            continue

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev KorQuADDataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark KorQuADDataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
