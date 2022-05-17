length = 24
n_gram = True
val_rate = 0.2
num_words = 25
batch_size = 1024
epochs = 50
val_threshold = 0.5     # 検証用データに対する陽性・陰性の閾値
threshold = 0.99        # モデルの評価を行うときの陽性・陰性の閾値
head_num = 8            # Transformerの並列化に関するパラメータ
dropout_rate = 0.1
hopping_num = 4         # Multi-Head Attentionを施す回数
hidden_dim = 128        # 単語ベクトルの次元数
lr = 0.001              # 学習率
beta = 0.5				# Fベータスコアの引数

fasta_dir = data/interim
processed_dir = data/processed
tfrecord_dir = data/tfrecord
model_dir = models
result_dir = reports/result

INPUT = $(wildcard data/interim/*.fasta)
PROCESSED = $(INPUT:data/interim/%.fasta=data/processed/%.pickle)
VOCAB_FILE = references/vocab.pickle
TRAIN_TFRECORD = $(tfrecord_dir)/train_dataset.tfrecord
TEST_TFRECORD = $(tfrecord_dir)/test_dataset.tfrecord
EVAL_TFRECORD = $(INPUT:data/interim/%.fasta=$(tfrecord_dir)/eval_%.tfrecord)
CLASS_WEIGHT = references/n_positive_negative.json
TRAINED_MODEL = models/saved_model.pb
RESULT = reports/result/evaluation.json
FALSE_POSITIVE = $(INPUT:data/interim/%.fasta=$(result_dir)/fp_%.txt)


all: $(RESULT) $(FALSE_POSITIVE)

$(PROCESSED): $(INPUT)
	python3 -m src.make_dataset $(length) $(fasta_dir) $(processed_dir) \
		--n_gram $(n_gram)

$(TRAIN_TFRECORD): $(PROCESSED) $(VOCAB_FILE)
	python3 src/convert_dataset.py $(processed_dir) $(TRAIN_TFRECORD) \
		$(TEST_TFRECORD) $(tfrecord_dir) $(VOCAB_FILE) $(CLASS_WEIGHT) \
		--val_rate $(val_rate)

$(TEST_TFRECORD): $(PROCESSED) $(VOCAB_FILE)
	python3 src/convert_dataset.py $(processed_dir) $(TRAIN_TFRECORD) \
		$(TEST_TFRECORD) $(tfrecord_dir) $(VOCAB_FILE) $(CLASS_WEIGHT) \
		--val_rate $(val_rate)

$(CLASS_WEIGHT): $(PROCESSED) $(VOCAB_FILE)
	python3 src/convert_dataset.py $(processed_dir) $(TRAIN_TFRECORD) \
		$(TEST_TFRECORD) $(tfrecord_dir) $(VOCAB_FILE) $(CLASS_WEIGHT) \
		--val_rate $(val_rate)

$(EVAL_TFRECORD): $(PROCESSED) $(VOCAB_FILE)
	python3 src/convert_dataset.py $(processed_dir) $(TRAIN_TFRECORD) \
		$(TEST_TFRECORD) $(tfrecord_dir) $(VOCAB_FILE) $(CLASS_WEIGHT) \
		--val_rate $(val_rate)

$(VOCAB_FILE): $(PROCESSED)
	python3 src/fit_vocab.py $(processed_dir) $(VOCAB_FILE) $(num_words)

$(TRAINED_MODEL): $(TRAIN_TFRECORD) $(TEST_TFRECORD) $(CLASS_WEIGHT)
	python3 src/train_model.py $(length) $(num_words) $(batch_size) \
		$(epochs) $(hopping_num) $(head_num) $(hidden_dim) $(dropout_rate) \
		$(lr) $(val_threshold) $(TRAIN_TFRECORD) $(TEST_TFRECORD) \
		$(model_dir) $(CLASS_WEIGHT)

$(RESULT): $(EVAL_TFRECORD) $(TRAINED_MODEL)
	python3 src/predict_model.py $(length) $(batch_size) $(num_words) \
		$(hopping_num) $(head_num) $(hidden_dim) $(dropout_rate) \
		$(lr) $(beta) $(model_dir) $(tfrecord_dir) $(TEST_TFRECORD) \
		$(VOCAB_FILE) $(RESULT) $(result_dir)

$(FALSE_POSITIVE): $(EVAL_TFRECORD) $(TRAINED_MODEL)
	python3 src/predict_model.py $(length) $(batch_size) $(num_words) \
		$(hopping_num) $(head_num) $(hidden_dim) $(dropout_rate) \
		$(lr) $(beta) $(model_dir) $(tfrecord_dir) $(TEST_TFRECORD) \
		$(VOCAB_FILE) $(RESULT) $(result_dir)
