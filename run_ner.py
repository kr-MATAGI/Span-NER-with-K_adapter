import json
import os
import torch
import numpy as np
import glob
import re

from attrdict import AttrDict
from tqdm import tqdm
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from transformers import get_linear_schedule_with_warmup, ElectraConfig, ElectraTokenizer

from model.adapter import AdapterModel
from model.pretrained_model import PretrainedModel

from tag_def import ETRI_TAG
from datasets import SpanNERDataset, AdapterDatasets
from utils import (
    print_parameters, load_corpus_span_ner_npy, load_adapter_npy_datasets, load_ner_config_and_model,
    init_logger, set_seed, f1_pre_rec, show_ner_report, load_adapter_config
)

# Global Variable
logger = init_logger()

#===============================================================
def evaluate(args, model, eval_dataset, mode, global_step=None, train_epoch=0):
#===============================================================
    results = {}

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval !
    if None != global_step:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))

    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    eval_pbar = tqdm(eval_dataloader)
    for batch in eval_pbar:
        model.eval()

        with torch.no_grad():
            label_ids = batch["label_ids"]
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                "label_ids": batch["label_ids"].to(args.device),
                "pos_ids": batch["pos_ids"].to(args.device),

                "all_span_idxs_ltoken": batch["all_span_idx"].to(args.device),
                "all_span_lens": batch["all_span_len"].to(args.device),
                "real_span_mask_ltoken": batch["real_span_mask"].to(args.device),
                "span_only_label": batch["span_only_label"].to(args.device),
                "mode": "eval"
            }

            loss, predict = model(**inputs)
            eval_loss += loss.mean().item()

        nb_eval_steps += 1
        eval_pbar.set_description("Eval Loss - %.04f" % (eval_loss / nb_eval_steps))

        if preds is None:
            preds = np.array(predict)
            out_label_ids = label_ids.numpy()
        else:
            preds = np.append(preds, np.array(predict), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.numpy(), axis=0)

    logger.info("  Eval End !")
    eval_pbar.close()

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }

    labels = ETRI_TAG.keys()
    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    ignore_index = torch.nn.CrossEntropyLoss().ignore_index
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != ignore_index:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(preds[i][j])
    result = f1_pre_rec(out_label_list, preds_list, is_ner=True)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))
        logger.info("\n" + show_ner_report(out_label_list, preds_list))  # Show report for each tag result
        f_w.write("\n" + show_ner_report(out_label_list, preds_list))

    return results

#=======================================================
def train(args, model, train_dataset, dev_dataset):
#=======================================================
    train_data_len = len(train_dataset)

    pretrained_model = model[0]
    adapter_model = model[1]

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_data_len // args.gradient_accumulation_steps) + 1
    else:
        t_total = (train_data_len // args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in adapter_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in adapter_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # @NOTE: optimizer에 설정된 learning_rate까지 선형으로 감소시킨다. (스케줄러)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    train_sampler = RandomSampler(train_dataset)
    pretrained_model.zero_grad()
    adapter_model.zero_grad()
    for epoch in range(args.num_train_epochs):
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        pbar = tqdm(train_dataloader)

        for step, batch in enumerate(pbar):
            pretrained_model.eval()
            adapter_model.train()

            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                "label_ids": batch["label_ids"].to(args.device),
            }

            pretrained_model_outputs = pretrained_model(**inputs)
            outputs = adapter_model(pretrained_model_outputs, **inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                    (len(train_dataloader) <= args.gradient_accumulation_steps and \
                     (step + 1) == len(train_dataloader)
                    ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()

                global_step += 1

                pbar.set_description("Train Loss - %.04f" % (tr_loss / global_step))

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save samples checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        adapter_model.module if hasattr(adapter_model, "module") else adapter_model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving samples checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        logger.info("  Epoch Done= %d", epoch + 1)
        pbar.close()

    return global_step, tr_loss / global_step

#=======================================================
def main():
#=======================================================
    print(f"[run_ner][main] Start - Main")

    # For SpanTags
    span_tag_dict = {
        'O': 0, 'FD': 1, 'EV': 2, 'DT': 3, 'TI': 4, 'MT': 5,
        'AM': 6, 'LC': 7, 'CV': 8, 'PS': 9, 'TR': 10,
        'TM': 11, 'AF': 12, 'PT': 13, 'OG': 14, 'QT': 15
    }

    ## INIT
    # Load Config, Model
    config_file_path = "./CONFIG.json"

    args = None
    with open(config_file_path) as config_file:
        args = AttrDict(json.load(config_file))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    plm_tokenizer = ElectraTokenizer.from_pretrained("./utils/tokenizer")
    plm_model = PretrainedModel(plm_tokenizer)

    with open("./corpus/klue_re/relation_list.json", "r", encoding="utf-8") as f:
        relation_class = json.load(f)["relations"]
        label_map = {label: i for i, label in enumerate(relation_class)}

    adapter_model = AdapterModel(args, plm_model.config, num_labels=len(label_map.keys()))

    plm_model.to(args.device)
    adapter_model.to(args.device)
    model = (plm_model, adapter_model)

    logger.info(f"[run_ner][__main__] model: {args.model_name_or_path}")
    logger.info(f"Training/Evaluation parameters")
    print_parameters(args, logger)

    # Set seed
    set_seed(args)
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    # Load npy
    adapter_dataset_path = "./corpus/npy/klue_re/"
    train_input_ids, train_attn_mask, train_tok_type_ids, train_label_ids = \
        load_adapter_npy_datasets(src_path=adapter_dataset_path, mode="train")
    dev_input_ids, dev_attn_mask, dev_tok_type_ids, dev_label_ids = \
        load_adapter_npy_datasets(src_path=adapter_dataset_path, mode="dev")

    # Make Datasets
    train_dataset = AdapterDatasets(input_ids=train_input_ids, label_ids=train_label_ids,
                                    attn_mask=train_attn_mask, tok_type_ids=train_tok_type_ids)
    dev_dataset = AdapterDatasets(input_ids=dev_input_ids, label_ids=dev_label_ids,
                                  attn_mask=dev_attn_mask, tok_type_ids=dev_tok_type_ids)

    # Train
    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset)
        logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

    results = {}
    if args.do_eval:
        checkpoints = list(os.path.dirname(c) for c in
                           sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True),
                                  key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1]))

        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logger.info("transformers.configuration_utils")
            logger.info("transformers.modeling_utils")
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = SpanNER.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            if len(checkpoints) > 1:
                for key in sorted(results.keys(), key=lambda key_with_step: (
                        "".join(re.findall(r'[^_]+_', key_with_step)),
                        int(re.findall(r"_\d+", key_with_step)[-1][1:])
                )):
                    f_w.write("{} = {}\n".format(key, str(results[key])))
            else:
                for key in sorted(results.keys()):
                    f_w.write("{} = {}\n".format(key, str(results[key])))

if "__main__" == __name__:
    main()