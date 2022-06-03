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
seed = 1				# データセットをシャッフルするときのseed値

viruses = $(notdir $(basename $(wildcard data/interim/*.fasta)))

fasta_dir = data/interim
processed_dir = data/processed
tfrecord_dir = data/tfrecord
eval_tfrecord_dir = $(tfrecord_dir)/eval
model_dir = models
result_dir = reports/result
false_positive_dir = reports/result/false_positive

INPUT = $(foreach virus,$(viruses),data/interim/$(virus).fasta)
PROCESSED = $(foreach virus,$(viruses),data/processed/$(virus)/.finish)
VOCAB_FILE = references/vocab.pickle
TRAIN_TFRECORD = $(tfrecord_dir)/train_dataset.tfrecord
TEST_TFRECORD = $(tfrecord_dir)/test_dataset.tfrecord
EVAL_TFRECORD_DIR = $(foreach virus,$(viruses),$(eval_tfrecord_dir)/$(virus))
x = $(foreach virus,$(viruses),$(wildcard $(processed_dir)/$(virus)/*.pickle))
EVAL_TFRECORD = $(x:$(processed_dir)/%.pickle=$(eval_tfrecord_dir)/%.tfrecord)
CLASS_WEIGHT = references/n_positive_negative.json
TRAINED_MODEL = models/saved_model.pb
RESULT = reports/result/evaluation.csv
FALSE_POSITIVE_DIR = $(foreach virus,$(viruses),\
	$(false_positive_dir)/$(virus))
FALSE_POSITIVE = $(false_positive_dir)/.finish


all: $(RESULT) $(FALSE_POSITIVE)

$(PROCESSED): $(INPUT)
	mkdir -p $(@D)
	python3 -m src.make_dataset $(length) \
		$(foreach virus,$(viruses),$(findstring $(virus),$@)) \
		$(fasta_dir) $(@D) --n_gram $(n_gram)
	touch $@

$(VOCAB_FILE): $(PROCESSED)
	python3 src/fit_vocab.py $(processed_dir) $(VOCAB_FILE) $(num_words)

$(TRAIN_TFRECORD): $(PROCESSED) $(VOCAB_FILE) $(EVAL_TFRECORD_DIR)
	python3 src/convert_dataset.py $(processed_dir) $(TRAIN_TFRECORD) \
		$(TEST_TFRECORD) $(eval_tfrecord_dir) $(VOCAB_FILE) $(CLASS_WEIGHT) \
		--val_rate $(val_rate) --seed $(seed)

$(TEST_TFRECORD): $(PROCESSED) $(VOCAB_FILE) $(EVAL_TFRECORD_DIR)
	python3 src/convert_dataset.py $(processed_dir) $(TRAIN_TFRECORD) \
		$(TEST_TFRECORD) $(eval_tfrecord_dir) $(VOCAB_FILE) $(CLASS_WEIGHT) \
		--val_rate $(val_rate) --seed $(seed)

$(EVAL_TFRECORD): $(PROCESSED) $(VOCAB_FILE) $(EVAL_TFRECORD_DIR)
	python3 src/convert_dataset.py $(processed_dir) $(TRAIN_TFRECORD) \
$(TEST_TFRECORD) $(eval_tfrecord_dir) $(VOCAB_FILE) $(CLASS_WEIGHT) \
		--val_rate $(val_rate) --seed $(seed)

$(CLASS_WEIGHT): $(PROCESSED) $(VOCAB_FILE) $(EVAL_TFRECORD_DIR)
	python3 src/convert_dataset.py $(processed_dir) $(TRAIN_TFRECORD) \
		$(TEST_TFRECORD) $(eval_tfrecord_dir) $(VOCAB_FILE) $(CLASS_WEIGHT) \
		--val_rate $(val_rate) --seed $(seed)

$(FALSE_POSITIVE_DIR): $(EVAL_TFRECORD_DIR)
	mkdir -p $@

$(RESULT): $(EVAL_TFRECORD) $(TRAINED_MODEL) $(FALSE_POSITIVE_DIR)
	python3 src/predict_model.py $(length) $(batch_size) $(num_words) \
		$(hopping_num) $(head_num) $(hidden_dim) $(dropout_rate) \
		$(lr) $(beta) $(model_dir) $(eval_tfrecord_dir) $(TEST_TFRECORD) \
		$(VOCAB_FILE) $(RESULT) $(false_positive_dir)
	touch $(FALSE_POSITIVE)

$(FALSE_POSITIVE): $(EVAL_TFRECORD) $(TRAINED_MODEL) $(FALSE_POSITIVE_DIR)
	python3 src/predict_model.py $(length) $(batch_size) $(num_words) \
		$(hopping_num) $(head_num) $(hidden_dim) $(dropout_rate) \
		$(lr) $(beta) $(model_dir) $(eval_tfrecord_dir) $(TEST_TFRECORD) \
		$(VOCAB_FILE) $(RESULT) $(false_positive_dir)
	touch $(FALSE_POSITIVE)
