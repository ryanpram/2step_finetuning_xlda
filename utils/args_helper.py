from utils.data_utils import AspectExtractionDataset, AspectExtractionDataLoader
from utils.data_utils import NerGritDataset, NerProsaDataset, NerDataLoader
from utils.data_utils import PosTagIdnDataset, PosTagProsaDataset, PosTagDataLoader
from utils.data_utils import EmotionDetectionDataset, EmotionDetectionDataLoader
from utils.data_utils import EntailmentDataset, EntailmentDataLoader
from utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader
from utils.data_utils import KeywordExtractionDataset, KeywordExtractionDataLoader
from utils.data_utils import NewsCategorizationDataset, NewsCategorizationDataLoader
from utils.data_utils import QAFactoidDataset, QAFactoidDataLoader
#from utils.data_utils import SquadLikeDataset
from utils.data_utils import AspectBasedSentimentAnalysisAiryDataset, AspectBasedSentimentAnalysisProsaDataset, AspectBasedSentimentAnalysisDataLoader

from utils.functions import WordSplitTokenizer
from utils.metrics import emotion_detection_metrics_fn, aspect_extraction_metrics_fn, ner_metrics_fn, pos_tag_metrics_fn, entailment_metrics_fn, document_sentiment_metrics_fn, keyword_extraction_metrics_fn, news_categorization_metrics_fn, qa_factoid_metrics_fn, absa_metrics_fn
from utils.forward_fn import forward_sequence_classification, forward_word_classification, forward_sequence_multi_classification
from torch.utils.data import TensorDataset, DataLoader

from nltk.tokenize import TweetTokenizer
from argparse import ArgumentParser
import json

###
# args functions
###
def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.keys():
        if opts[key]:
            print('{:>30}: {:<50}'.format(key, opts[key]).center(80))
    print('=' * 80)

def save_opts(opts,model_dir):
    """save args to txt file
    """
    with open(model_dir + '/args.txt', 'w') as f:
        json.dump(opts.__dict__, f, indent=2)
    print("Args saved to ",model_dir + '/args.txt')
    return True

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--model_dir", type=str, default="save/", help="Model directory")
    parser.add_argument("--dataset", type=str, default='emotion-twitter', help="Choose between emotion-twitter, absa-airy, term-extraction-airy, ner-grit, pos-idn, entailment-ui, doc-sentiment-prosa, keyword-extraction-prosa, qa-factoid-itb, news-category-prosa, ner-prosa, pos-prosa")
    parser.add_argument("--model_checkpoint", type=str, default="bert-base-multilingual-uncased", help="Path, url or short name of the model")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max number of tokens")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=10.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--device", type=str, default='cuda', help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--step_size", type=int, default=1, help="Step size")
    parser.add_argument("--early_stop", type=int, default=3, help="Step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma")
    parser.add_argument("--debug", action='store_true', help="debugging mode")
    parser.add_argument("--force", action='store_true', help="force to rewrite experiment folder")
    parser.add_argument("--eval_only", action='store_true', help="for skipping training process")
    parser.add_argument("--no_special_token", action='store_true', help="not adding special token as the input")
    parser.add_argument("--lower", action='store_true', help="lower case")
    parser.add_argument('--save_history_steps', type=int, default=50, help="Save Training history every X updates steps.")
    parser.add_argument('--save_eval_history_steps', type=int, default=50, help="Save Eval history every X updates steps.")
    parser.add_argument('--dropout_prob', type=float, default=0.1, help="Dropout Probability value.")
    parser.add_argument('--data_type', type=str, default='original', help="For determine dataset.")
    parser.add_argument('--model_type', type=str, default='xlm-roberta', help="For determine dataset.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = vars(parser.parse_args())
    print_opts(args)
    return args

def get_eval_parser():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--model_dir", type=str, default="./save", help="Model directory")
    parser.add_argument("--dataset", type=str, default='emotion-twitter', help="Choose between emotion-twitter, absa-airy, term-extraction-airy, ner-grit, pos-idn, entailment-ui, doc-sentiment-prosa, keyword-extraction-prosa, qa-factoid-itb, news-category-prosa, ner-prosa, pos-prosa")
    parser.add_argument("--model_type", type=str, default="bert-base-multilingual-uncased", help="Type of the model")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max number of tokens")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--debug", action='store_true', help="debugging mode")
    parser.add_argument("--no_special_token", action='store_true', help="not adding special token as the input")
    parser.add_argument("--lower", action='store_true', help="lower case")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default='cuda', help="Device (cuda or cpu)")

    args = vars(parser.parse_args())
    print_opts(args)
    return args

#TODO: Need to change it into a json or something else that are easily extendable
def append_dataset_args(args):      
    if args['dataset'] == "emotion-twitter":
        args['task'] = 'sequence_classification'
        args['num_labels'] = EmotionDetectionDataset.NUM_LABELS
        args['dataset_class'] = EmotionDetectionDataset
        args['dataloader_class'] = EmotionDetectionDataLoader
        args['forward_fn'] = forward_sequence_classification
        args['metrics_fn'] = emotion_detection_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/emot_emotion-twitter/train_preprocess.csv'
        args['valid_set_path'] = './dataset/emot_emotion-twitter/valid_preprocess.csv'
        args['test_set_path'] = './dataset/emot_emotion-twitter/test_preprocess_masked_label.csv'
        args['vocab_path']  = "./dataset/emot_emotion-twitter/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_emotion-twitter_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_emotion-twitter_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "absa-airy":
        args['task'] = 'multi_label_classification'
        args['num_labels'] = AspectBasedSentimentAnalysisAiryDataset.NUM_LABELS
        args['dataset_class'] = AspectBasedSentimentAnalysisAiryDataset
        args['dataloader_class'] = AspectBasedSentimentAnalysisDataLoader
        args['forward_fn'] = forward_sequence_multi_classification
        args['metrics_fn'] = absa_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/hoasa_absa-airy/train_preprocess.csv'
        args['valid_set_path'] = './dataset/hoasa_absa-airy/valid_preprocess.csv'
        args['test_set_path'] = './dataset/hoasa_absa-airy/test_preprocess_masked_label.csv'
        args['vocab_path'] = "./dataset/hoasa_absa-airy/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_absa-airy_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_absa-airy_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "term-extraction-airy":
        args['task'] = 'token_classification'
        args['num_labels'] = AspectExtractionDataset.NUM_LABELS
        args['dataset_class'] = AspectExtractionDataset
        args['dataloader_class'] = AspectExtractionDataLoader
        args['forward_fn'] = forward_word_classification
        args['metrics_fn'] = aspect_extraction_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/terma_term-extraction-airy/train_preprocess.txt'
        args['valid_set_path'] = './dataset/terma_term-extraction-airy/valid_preprocess.txt'
        args['test_set_path'] = './dataset/terma_term-extraction-airy/test_preprocess_masked_label.txt'
        args['vocab_path'] = "./dataset/terma_term-extraction-airy/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_term-extraction-airy_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_term-extraction-airy_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "ner-grit":
        args['task'] = 'token_classification'
        args['num_labels'] = NerGritDataset.NUM_LABELS
        args['dataset_class'] = NerGritDataset
        args['dataloader_class'] = NerDataLoader
        args['forward_fn'] = forward_word_classification
        args['metrics_fn'] = ner_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/nergrit_ner-grit/train_preprocess.txt'
        args['valid_set_path'] = './dataset/nergrit_ner-grit/valid_preprocess.txt'
        args['test_set_path'] = './dataset/nergrit_ner-grit/test_preprocess_masked_label.txt'
        args['vocab_path'] = "./dataset/nergrit_ner-grit/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_ner-grit_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_ner-grit_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "pos-idn":
        args['task'] = 'token_classification'
        args['num_labels'] = PosTagIdnDataset.NUM_LABELS
        args['dataset_class'] = PosTagIdnDataset
        args['dataloader_class'] = PosTagDataLoader
        args['forward_fn'] = forward_word_classification
        args['metrics_fn'] = pos_tag_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/bapos_pos-idn/train_preprocess.txt'
        args['valid_set_path'] = './dataset/bapos_pos-idn/valid_preprocess.txt'
        args['test_set_path'] = './dataset/bapos_pos-idn/test_preprocess_masked_label.txt'
        args['vocab_path'] = "./dataset/bapos_pos-idn/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_pos-idn_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_pos-idn_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "entailment-ui":
        args['task'] = 'sequence_classification'
        args['num_labels'] = EntailmentDataset.NUM_LABELS
        args['dataset_class'] = EntailmentDataset
        args['dataloader_class'] = EntailmentDataLoader
        args['forward_fn'] = forward_sequence_classification
        args['metrics_fn'] = entailment_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/wrete_entailment-ui/train_preprocess.csv'
        args['valid_set_path'] = './dataset/wrete_entailment-ui/valid_preprocess.csv'
        args['test_set_path'] = './dataset/wrete_entailment-ui/test_preprocess_masked_label.csv'
        args['vocab_path'] = "./dataset/wrete_entailment-ui/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_entailment-ui_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_entailment-ui_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "doc-sentiment-prosa":
        args['task'] = 'sequence_classification'
        args['num_labels'] = DocumentSentimentDataset.NUM_LABELS
        args['dataset_class'] = DocumentSentimentDataset
        args['dataloader_class'] = DocumentSentimentDataLoader
        args['forward_fn'] = forward_sequence_classification
        args['metrics_fn'] = document_sentiment_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv'
        args['valid_set_path'] = './dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv'
        args['test_set_path'] = './dataset/smsa_doc-sentiment-prosa/test_preprocess_masked_label.tsv'
        args['vocab_path'] = "./dataset/smsa_doc-sentiment-prosa/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_doc-sentiment-prosa_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_doc-sentiment-prosa_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "keyword-extraction-prosa":
        args['task'] = 'token_classification'
        args['num_labels'] = KeywordExtractionDataset.NUM_LABELS
        args['dataset_class'] = KeywordExtractionDataset
        args['dataloader_class'] = KeywordExtractionDataLoader
        args['forward_fn'] = forward_word_classification
        args['metrics_fn'] = keyword_extraction_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/keps_keyword-extraction-prosa/train_preprocess.txt'
        args['valid_set_path'] = './dataset/keps_keyword-extraction-prosa/valid_preprocess.txt'
        args['test_set_path'] = './dataset/keps_keyword-extraction-prosa/test_preprocess_masked_label.txt'
        args['vocab_path'] = "./dataset/keps_keyword-extraction-prosa/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_keyword-extraction-prosa_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_keyword-extraction-prosa_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = WordSplitTokenizer
    elif args['dataset'] == "qa-factoid-itb":
        args['task'] = 'token_classification'
        args['num_labels'] = QAFactoidDataset.NUM_LABELS
        args['dataset_class'] = QAFactoidDataset
        args['dataloader_class'] = QAFactoidDataLoader
        args['forward_fn'] = forward_word_classification
        args['metrics_fn'] = qa_factoid_metrics_fn
        args['valid_criterion'] = 'F1'

        if args['data_type'] == 'combined':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/combined_train_data.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
        elif args['data_type'] == 'english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_preprocess.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
        elif args['data_type'] == 'original':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/train_preprocess.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
        elif args['data_type'] == 'chinese-only':
            args['train_set_path'] = './dataset/crmc/train_preprocessed.csv'
            args['valid_set_path'] = './dataset/crmc/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/crmc/test_preprocessed.csv'
        elif args['data_type'] == 'chinese-combined':
            args['train_set_path'] = './dataset/crmc/combined_zh_preprocessed.csv'
            args['valid_set_path'] = './dataset/crmc/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/crmc/test_preprocessed.csv'
        elif args['data_type'] == 'chinese-english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_preprocess.csv'
            args['valid_set_path'] = './dataset/crmc/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/crmc/test_preprocessed.csv'
        elif args['data_type'] == 'france-only':
            args['train_set_path'] = './dataset/fquad/train_preprocessed.csv'
            args['valid_set_path'] = './dataset/fquad/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/fquad/test_preprocessed.csv'
        elif args['data_type'] == 'france-combined':
            args['train_set_path'] = './dataset/fquad/combined_fr_preprocessed.csv'
            args['valid_set_path'] = './dataset/fquad/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/fquad/test_preprocessed.csv'
        elif args['data_type'] == 'france-english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_preprocess.csv'
            args['valid_set_path'] = './dataset/fquad/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/fquad/test_preprocessed.csv'
        elif args['data_type'] == 'viet-only':
            args['train_set_path'] = './dataset/ViQuAD/train_preprocessed.csv'
            args['valid_set_path'] = './dataset/ViQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/ViQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'viet-combined':
            args['train_set_path'] = './dataset/ViQuAD/combined_vi_preprocessed.csv'
            args['valid_set_path'] = './dataset/ViQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/ViQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'viet-english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_preprocess.csv'
            args['valid_set_path'] = './dataset/ViQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/ViQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'viet-broken':
            args['train_set_path'] = './dataset/ViQuAD/train_sample_preprocessed.csv'
            args['valid_set_path'] = './dataset/ViQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/ViQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'korean-only':
            args['train_set_path'] = './dataset/KorQuAD/train_preprocessed.csv'
            args['valid_set_path'] = './dataset/KorQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/KorQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'korean-combined':
            args['train_set_path'] = './dataset/KorQuAD/combined_full_ver_kor_preprocessed.csv'
            args['valid_set_path'] = './dataset/KorQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/KorQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'korean-english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_unfiltered_preprocess.csv'
            args['valid_set_path'] = './dataset/KorQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/KorQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'test-indo':
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'

        # Translated data augmentation 
        elif args['data_type'] == 'combined-translated':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/combined_newsqa_translated_id.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
        elif args['data_type'] == 'newsqa-id-3022-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/newsqa/translated_final/newsqa_id_3022.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
        
        # Code switching data augmentation 
        elif args['data_type'] == 'combined-cs':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/newsqa/cs/trainid-combined-facqa.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
        elif args['data_type'] == 'newsqa-cs-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/newsqa/cs/trainid-v1.1.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'


        # args['test_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
        args['vocab_path'] = "./dataset/facqa_qa-factoid-itb/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_qa-factoid-itb_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_qa-factoid-itb_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "squad-like-dataset":
        args['task'] = 'question_answering'
        #args['dataset_class'] = SquadLikeDataset
        args['dataloader_class'] = QAFactoidDataLoader
        args['forward_fn'] = forward_word_classification
        args['metrics_fn'] = qa_factoid_metrics_fn
        args['valid_criterion'] = 'F1'

        if args['data_type'] == 'combined':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/combined_train_data.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
        elif args['data_type'] == 'english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_preprocess.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
        elif args['data_type'] == 'original':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/train_preprocess.csv'
            args['valid_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
            args['test_set_path'] = './dataset/facqa_qa-factoid-itb/test_preprocess_masked_label.csv'
        elif args['data_type'] == 'chinese-only':
            args['train_set_path'] = './dataset/crmc/train_preprocessed.csv'
            args['valid_set_path'] = './dataset/crmc/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/crmc/test_preprocessed.csv'
        elif args['data_type'] == 'chinese-combined':
            args['train_set_path'] = './dataset/crmc/combined_zh_preprocessed.csv'
            args['valid_set_path'] = './dataset/crmc/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/crmc/test_preprocessed.csv'
        elif args['data_type'] == 'chinese-english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_preprocess.csv'
            args['valid_set_path'] = './dataset/crmc/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/crmc/test_preprocessed.csv'
        elif args['data_type'] == 'france-only':
            args['train_set_path'] = './dataset/fquad/train_preprocessed.csv'
            args['valid_set_path'] = './dataset/fquad/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/fquad/test_preprocessed.csv'
        elif args['data_type'] == 'france-combined':
            args['train_set_path'] = './dataset/fquad/combined_fr_preprocessed.csv'
            args['valid_set_path'] = './dataset/fquad/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/fquad/test_preprocessed.csv'
        elif args['data_type'] == 'france-english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_preprocess.csv'
            args['valid_set_path'] = './dataset/fquad/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/fquad/test_preprocessed.csv'
        elif args['data_type'] == 'viet-only':
            args['train_set_path'] = './dataset/ViQuAD/train_preprocessed.csv'
            args['valid_set_path'] = './dataset/ViQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/ViQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'viet-combined':
            args['train_set_path'] = './dataset/ViQuAD/combined_vi_preprocessed.csv'
            args['valid_set_path'] = './dataset/ViQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/ViQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'viet-english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_preprocess.csv'
            args['valid_set_path'] = './dataset/ViQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/ViQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'viet-broken':
            args['train_set_path'] = './dataset/ViQuAD/train_sample_preprocessed.csv'
            args['valid_set_path'] = './dataset/ViQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/ViQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'korean-only':
            args['train_set_path'] = './dataset/KorQuAD/train_preprocessed.csv'
            args['valid_set_path'] = './dataset/KorQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/KorQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'korean-combined':
            args['train_set_path'] = './dataset/KorQuAD/combined_full_ver_kor_preprocessed.csv'
            args['valid_set_path'] = './dataset/KorQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/KorQuAD/test_preprocessed.csv'
        elif args['data_type'] == 'korean-english-only':
            args['train_set_path'] = './dataset/facqa_qa-factoid-itb/eng_train_unfiltered_preprocess.csv'
            args['valid_set_path'] = './dataset/KorQuAD/valid_preprocessed.csv'
            args['test_set_path'] = './dataset/KorQuAD/test_preprocessed.csv'



        # args['test_set_path'] = './dataset/facqa_qa-factoid-itb/valid_preprocess.csv'
        args['vocab_path'] = "./dataset/facqa_qa-factoid-itb/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_qa-factoid-itb_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_qa-factoid-itb_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    
    elif args['dataset'] == "ner-prosa":
        args['task'] = 'token_classification'
        args['num_labels'] = NerProsaDataset.NUM_LABELS
        args['dataset_class'] = NerProsaDataset
        args['dataloader_class'] = NerDataLoader
        args['forward_fn'] = forward_word_classification
        args['metrics_fn'] = ner_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/nerp_ner-prosa/train_preprocess.txt'
        args['valid_set_path'] = './dataset/nerp_ner-prosa/valid_preprocess.txt'
        args['test_set_path'] = './dataset/nerp_ner-prosa/test_preprocess_masked_label.txt'
        args['vocab_path'] = "./dataset/nerp_ner-prosa/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_ner-prosa_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_ner-prosa_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "pos-prosa":
        args['task'] = 'token_classification'
        args['num_labels'] = PosTagProsaDataset.NUM_LABELS
        args['dataset_class'] = PosTagProsaDataset
        args['dataloader_class'] = PosTagDataLoader
        args['forward_fn'] = forward_word_classification
        args['metrics_fn'] = pos_tag_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/posp_pos-prosa/train_preprocess.txt'
        args['valid_set_path'] = './dataset/posp_pos-prosa/valid_preprocess.txt'
        args['test_set_path'] = './dataset/posp_pos-prosa/test_preprocess_masked_label.txt'
        args['vocab_path'] = "./dataset/posp_pos-prosa/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_pos-prosa_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_pos-prosa_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    elif args['dataset'] == "absa-prosa":
        args['task'] = 'multi_label_classification'
        args['num_labels'] = AspectBasedSentimentAnalysisProsaDataset.NUM_LABELS
        args['dataset_class'] = AspectBasedSentimentAnalysisProsaDataset
        args['dataloader_class'] = AspectBasedSentimentAnalysisDataLoader
        args['forward_fn'] = forward_sequence_multi_classification
        args['metrics_fn'] = absa_metrics_fn
        args['valid_criterion'] = 'F1'
        args['train_set_path'] = './dataset/casa_absa-prosa/train_preprocess.csv'
        args['valid_set_path'] = './dataset/casa_absa-prosa/valid_preprocess.csv'
        args['test_set_path'] = './dataset/casa_absa-prosa/test_preprocess_masked_label.csv'
        args['vocab_path'] = "./dataset/casa_absa-prosa/vocab_uncased.txt"
        args['embedding_path'] = {
            'fasttext-cc-id-300-no-oov-uncased': './embeddings/fasttext-cc-id/cc.id.300_no-oov_absa-prosa_uncased.txt',
            'fasttext-4B-id-300-no-oov-uncased': './embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_absa-prosa_uncased.txt'
        }
        args['k_fold'] = 1
        args['word_tokenizer_class'] = TweetTokenizer
    else:
        raise ValueError(f'Unknown dataset name `{args["dataset"]}`')

    print(args)
    return args
