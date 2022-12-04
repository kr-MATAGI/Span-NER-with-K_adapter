import logging
import numpy as np
import torch
import random

from tag_def import NIKL_POS_TAG, MECAB_POS_TAG

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

# config, model
from transformers import ElectraConfig, AutoConfig, AutoModelForTokenClassification, ElectraForTokenClassification
from model.span_ner import ElectraSpanNER

#===============================================================
def print_parameters(args, logger):
#===============================================================
    logger.info(f"ckpt_dir: {args.ckpt_dir}")
    logger.info(f"output_dir: {args.output_dir}")

    logger.info(f"train_npy: {args.train_npy}")
    logger.info(f"dev_npy: {args.dev_npy}")
    logger.info(f"test_npy: {args.test_npy}")

    logger.info(f"evaluate_test_during_training: {args.evaluate_test_during_training}")
    logger.info(f"eval_all_checkpoints: {args.eval_all_checkpoints}")

    logger.info(f"save_optimizer: {args.save_optimizer}")
    logger.info(f"do_lower_case: {args.do_lower_case}")

    logger.info(f"do_train: {args.do_train}")
    logger.info(f"do_eval: {args.do_eval}")

    logger.info(f"max_seq_len: {args.max_seq_len}")
    logger.info(f"num_train_epochs: {args.num_train_epochs}")

    logger.info(f"weight_decay: {args.weight_decay}")
    logger.info(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    logger.info(f"adam_epsilon: {args.adam_epsilon}")
    logger.info(f"warmup_proportion: {args.warmup_proportion}")

    logger.info(f"max_steps: {args.max_steps}")
    logger.info(f"max_grad_norm: {args.max_grad_norm}")
    logger.info(f"seed: {args.seed}")

    logger.info(f"model_name_or_path: {args.model_name_or_path}")
    logger.info(f"train_batch_size: {args.train_batch_size}")
    logger.info(f"eval_batch_size: {args.eval_batch_size}")
    logger.info(f"learning_rate: {args.learning_rate}")

    logger.info(f"logging_steps: {args.logging_steps}")
    logger.info(f"save_steps: {args.save_steps}")

#===============================================================
def load_corpus_span_ner_npy(src_path: str, mode: str="train"):
#===============================================================
    root_path = "/".join(src_path.split("/")[:-1]) + "/" + mode

    input_token_attn_npy = np.load(src_path)
    label_ids = np.load(root_path + "_label_ids.npy")

    all_span_idx = np.load(root_path + "_all_span_idx.npy")
    all_span_len = np.load(root_path + "_all_span_len_list.npy")
    real_span_mask = np.load(root_path + "_real_span_mask_token.npy")
    span_only_label = np.load(root_path + "_span_only_label_token.npy")

    pos_ids = np.load(root_path + "_pos_ids.npy")

    print(f"{mode}.shape - dataset: {input_token_attn_npy.shape}, label_ids: {label_ids.shape}, pos_ids: {pos_ids.shape}"
          f"all_span_idx: {all_span_idx.shape}, all_span_len: {all_span_len.shape}, "
          f"real_span_mask: {real_span_mask.shape}, span_only_label: {span_only_label.shape}")

    return input_token_attn_npy, label_ids, all_span_idx, all_span_len, real_span_mask, span_only_label, pos_ids

#===============================================================
def load_corpus_npy_datasets(src_path: str, mode: str="train"):
#===============================================================
    root_path = "/".join(src_path.split("/")[:-1]) + "/" + mode

    dataset_npy = np.load(src_path)
    pos_ids_npy = np.load(root_path + "_pos_ids.npy")
    label_ids_npy = np.load(root_path + "_label_ids.npy")

    return dataset_npy, label_ids_npy, pos_ids_npy

#===============================================================
def init_logger():
# ===============================================================
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    return logger

#===============================================================
def set_seed(args):
#===============================================================
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" == args.device:
        torch.cuda.manual_seed_all(args.seed)

#===============================================================
def f1_pre_rec(labels, preds, is_ner=True):
#===============================================================
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds),
            "recall": seqeval_metrics.recall_score(labels, preds),
            "f1": seqeval_metrics.f1_score(labels, preds),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }

#===============================================================
def show_ner_report(labels, preds):
#===============================================================
    return seqeval_metrics.classification_report(labels, preds, digits=3)

#===============================================================
def load_ner_config_and_model(args, tag_dict):
#===============================================================
    config = None
    model = None

    # Config
    span_tag_list = tag_dict.keys()
    print("SPAN_TAG_DICT: ", span_tag_list)
    config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                           num_labels=len(span_tag_list),
                                           id2label={idx: label for label, idx in tag_dict.items()},
                                           label2id={label: idx for label, idx in tag_dict.items()})
    config.etri_tags = {label: i for i, label in enumerate(tag_dict.keys())}

    # Model
    model = ElectraSpanNER.from_pretrained(args.model_name_or_path, config=config)

    return config, model
