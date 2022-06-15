length = 30
n_gram = True
val_rate = 0.2
num_words = 25
batch_size = 1024
epochs = 50
threshold = 0.5         # 陽性・陰性の閾値
head_num = 8            # Transformerの並列化に関するパラメータ
dropout_rate = 0.239
hopping_num = 6         # Multi-Head Attentionを施す回数
hidden_dim = 736        # 単語ベクトルの次元数
lr = 1.52e-5            # 学習率
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
	python3 -m src.make_dataset \
		$(length) \
		$(foreach virus,$(viruses),$(findstring $(virus),$@)) \
		$(fasta_dir) \
		$(@D) \
		--n_gram $(n_gram)
	touch $@

$(VOCAB_FILE): $(PROCESSED)
	python3 src/fit_vocab.py \
		$(processed_dir) \
		$(VOCAB_FILE) \
		$(num_words)

$(TRAIN_TFRECORD) $(TEST_TFRECORD) $(EVAL_TFRECORD) $(CLASS_WEIGHT): \
		$(PROCESSED) $(VOCAB_FILE)
	mkdir -p $(EVAL_TFRECORD_DIR)
	python3 src/convert_dataset.py \
		$(processed_dir) \
		$(TRAIN_TFRECORD) \
		$(TEST_TFRECORD) \
		$(eval_tfrecord_dir) \
		$(VOCAB_FILE) \
		$(CLASS_WEIGHT) \
		--val_rate $(val_rate) \
		--seed $(seed)

$(TRAINED_MODEL): $(TRAIN_TFRECORD) $(TEST_TFRECORD) $(CLASS_WEIGHT)
	python3 src/train_model.py \
		$(length) \
		$(num_words) \
		$(batch_size) \
		$(epochs) \
		$(hopping_num) \
		$(head_num) \
		$(hidden_dim) \
		$(dropout_rate) \
		$(lr) \
		$(threshold) \
		$(TRAIN_TFRECORD) \
		$(TEST_TFRECORD) \
		$(model_dir) \
		$(CLASS_WEIGHT)

$(FALSE_POSITIVE_DIR): $(EVAL_TFRECORD_DIR)
	mkdir -p $@

$(RESULT) $(FALSE_POSITIVE): $(EVAL_TFRECORD) $(TRAINED_MODEL)
	python3 src/predict_model.py \
		$(length) \
		$(batch_size) \
		$(num_words) \
		$(hopping_num) \
		$(head_num) \
		$(hidden_dim) \
		$(dropout_rate) \
		$(lr) \
		$(beta) \
		$(threshold) \
		$(model_dir) \
		$(eval_tfrecord_dir) \
		$(VOCAB_FILE) \
		$(RESULT) \
		$(false_positive_dir)
	touch $(FALSE_POSITIVE)

clear:
	find data/processed/ | grep -v -x 'data/processed/' | \
		grep -v '.gitkeep' | xargs rm -rf
	find data/tfrecord/ | grep -v -x 'data/tfrecord/' | \
		grep -v '.gitkeep' | xargs rm -rf
	rm -f $(VOCAB_FILE)
	rm -f $(CLASS_WEIGHT)
	find models/ | grep -v -x 'models/' | grep -v '.gitkeep' | \
		xargs rm -rf
	rm -f $(RESULT)
	find reports/result/ | grep -v -x 'reports/result/' | \
		grep -v '.gitkeep' | xargs rm -rf
