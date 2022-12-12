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

from model.dp_apdater import DpAdapterModel
from model.pretrained_model import PretrainedModel

# from token_utils.metrics import klue_re_auprc

from tag_def import ETRI_TAG
from sklearn.metrics import f1_score
from datasets import DpAdapterDatasets
from utils import (
    print_parameters, load_dp_adapter_npy_datasets, init_logger, set_seed,
)

from typing import List, Tuple, Any
from definition.klue_dp import pos_labels, dep_labels

# Global Variable
logger = init_logger()
g_no_pos = False

#===============================================================
def dp_collate_fn(batch: List[Tuple]):
#===============================================================
    batch_size = len(batch)
    pos_padding_idx = None if g_no_pos else len(pos_labels)

    batch_input_ids = []
    batch_attention_masks = []
    batch_bpe_head_masks = []
    batch_bpe_tail_masks = []

    for batch_id in range(batch_size):
        (
            input_id,
            attention_mask,
            bpe_head_mask,
            bpe_tail_mask,
            _,
            _,
            _,
        ) = batch[batch_id]
        batch_input_ids.append(input_id)
        batch_attention_masks.append(attention_mask)
        batch_bpe_head_masks.append(bpe_head_mask)
        batch_bpe_tail_masks.append(bpe_tail_mask)

    # 2. build inputs : packing tensors
    # 나는 밥을 먹는다. => [CLS] 나 ##는 밥 ##을 먹 ##는 ##다 . [SEP]
    # input_id : [2, 717, 2259, 1127, 2069, 1059, 2259, 2062, 18, 3, 0, 0, ...]
    # bpe_head_mask : [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, ...] (indicate word start (head) idx)
    input_ids = torch.stack(batch_input_ids)
    attention_masks = torch.stack(batch_attention_masks)
    bpe_head_masks = torch.stack(batch_bpe_head_masks)
    bpe_tail_masks = torch.stack(batch_bpe_tail_masks)
    # 3. token_to_words : set in-batch max_word_length
    max_word_length = max(torch.sum(bpe_head_masks, dim=1)).item()
    # 3. token_to_words : placeholders
    head_ids = torch.zeros(batch_size, max_word_length).long()
    type_ids = torch.zeros(batch_size, max_word_length).long()
    pos_ids = torch.zeros(batch_size, max_word_length + 1).long()
    mask_e = torch.zeros(batch_size, max_word_length + 1).long()
    # 3. token_to_words : head_ids, type_ids, pos_ids, mask_e, mask_d
    for batch_id in range(batch_size):
        (
            _,
            _,
            bpe_head_mask,
            _,
            token_head_ids,
            token_type_ids,
            token_pos_ids,
        ) = batch[batch_id]
        # head_id : [1, 3, 5] (prediction candidates)
        # token_head_ids : [-1, 3, -1, 3, -1, 0, -1, -1, -1, .-1, ...] (ground truth head ids)
        head_id = [i for i, token in enumerate(bpe_head_mask) if token == 1]
        word_length = len(head_id)
        head_id.extend([0] * (max_word_length - word_length))
        head_ids[batch_id] = token_head_ids[head_id]
        type_ids[batch_id] = token_type_ids[head_id]
        if not g_no_pos:
            pos_ids[batch_id][0] = torch.tensor(pos_padding_idx)
            pos_ids[batch_id][1:] = token_pos_ids[head_id]
            pos_ids[batch_id][int(torch.sum(bpe_head_mask)) + 1 :] = torch.tensor(pos_padding_idx)
        mask_e[batch_id] = torch.LongTensor([1] * (word_length + 1) + [0] * (max_word_length - word_length))
    mask_d = mask_e[:, 1:]
    # 4. pack everything
    masks = (attention_masks, bpe_head_masks, bpe_tail_masks, mask_e, mask_d)
    ids = (head_ids, type_ids, pos_ids)

    return input_ids, masks, ids, max_word_length

#===============================================================
def evaluate(args, model, eval_dataset, mode, global_step=None, train_epoch=0):
#===============================================================
    pretrained_model = model[0]
    adapter_model = model[1]

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
    prediction = []
    gold_result = []

    eval_pbar = tqdm(eval_dataloader)
    for batch in eval_pbar:
        pretrained_model.eval()
        adapter_model.eval()

        with torch.no_grad():
            label_ids = batch["label_ids"]
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                "label_ids": batch["label_ids"].to(args.device),
                "subj_start_id": batch["subj_start_id"].to(args.device),
                "obj_start_id": batch["obj_start_id"].to(args.device)
            }

            pretrained_model_outputs = pretrained_model(**inputs)
            outputs = adapter_model(pretrained_model_outputs, **inputs)

            tmp_eval_loss, logits = outputs[:2]
            preds = logits.argmax(dim=1)
            prediction += preds.tolist()
            gold_result += inputs['label_ids'].tolist()
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        eval_pbar.set_description("Eval Loss - %.04f" % (eval_loss / nb_eval_steps))
    # end loop, batch

    micro_F1 = f1_score(y_true=gold_result, y_pred=prediction, average='micro')
    macro_F1 = f1_score(y_true=gold_result, y_pred=prediction, average='macro')
    # auc_score = klue_re_auprc(probs=logits.detach().cpu(), labels=inputs["label_ids"].detach().cpu())

    logger.info("The micro_f1 on dev dataset: %f", micro_F1)
    logger.info("The macro_f1 on dev dataset: %f", macro_F1)
    # logger.info("The AUC on dev dataset: %f", auc_score)

    results['micro_F1'] = micro_F1
    results['macro_F1'] = macro_F1
    results['loss'] = eval_loss

    output_eval_file = os.path.join(args.output_dir, args.ckpt_dir + "_eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results  *****")
        for key in sorted(results.keys()):
            # logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))

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
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=dp_collate_fn)
        pbar = tqdm(train_dataloader)

        for step, batch in enumerate(pbar):
            pretrained_model.eval()
            adapter_model.train()

            input_ids, masks, ids, max_word_length = batch
            attention_mask, bpe_head_mask, bpe_tail_mask, mask_e, mask_d = masks
            head_ids, type_ids, pos_ids = ids

            batch_size = head_ids.size()[0]
            batch_index = torch.arange(0, int(batch_size)).long()

            inputs = {
                "input_ids": input_ids.to(args.device),
                "attention_mask": attention_mask.to(args.device),
                "bpe_head_mask": bpe_head_mask.to(args.device),
                "bpe_tail_mask": bpe_tail_mask.to(args.device),
                "max_word_length": max_word_length,
                "batch_index": batch_index.to(args.device),
                "mask_e": mask_e.to(args.device),
                "mask_d": mask_d.to(args.device),
                "head_ids": head_ids.to(args.device),
                "pos_ids": pos_ids.to(args.device),
                "type_ids": type_ids.to(args.device)
            }

            pretrained_model_outputs = pretrained_model(input_ids=inputs["input_ids"],
                                                        attention_mask=inputs["attention_mask"])
            loss = adapter_model(pretrained_model_outputs, **inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                    (len(train_dataloader) <= args.gradient_accumulation_steps and \
                     (step + 1) == len(train_dataloader)
                    ):
                torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), args.max_grad_norm)

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

    plm_tokenizer = ElectraTokenizer.from_pretrained("token_utils/tokenizer")
    plm_model = PretrainedModel(plm_tokenizer)
    adapter_model = DpAdapterModel(args, plm_model.config, len(pos_labels), len(dep_labels))

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
    adapter_dataset_path = "./corpus/npy/klue_dp/"
    train_input_ids_npy, train_attn_mask_npy, train_bpe_head_mask, train_bpe_tail_mask, train_head_ids, train_dep_ids, train_pos_ids = \
        load_dp_adapter_npy_datasets(src_path=adapter_dataset_path, mode="train")
    dev_input_ids_npy, dev_attn_mask_npy, dev_bpe_head_mask, dev_bpe_tail_mask, dev_head_ids, dev_dep_ids, dev_pos_ids = \
        load_dp_adapter_npy_datasets(src_path=adapter_dataset_path, mode="dev")

    # Make Datasets
    train_dataset = DpAdapterDatasets(input_ids=train_input_ids_npy, attn_mask=train_attn_mask_npy,
                                      bpe_head_mask=train_bpe_head_mask, bpe_tail_mask=train_bpe_tail_mask,
                                      dep_ids=train_dep_ids, head_ids=train_head_ids, pos_ids=train_pos_ids)
    dev_dataset = DpAdapterDatasets(input_ids=dev_input_ids_npy, attn_mask=dev_attn_mask_npy,
                                    bpe_head_mask=dev_bpe_head_mask, bpe_tail_mask=dev_bpe_tail_mask,
                                    dep_ids=dev_dep_ids, head_ids=dev_head_ids, pos_ids=dev_pos_ids)

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
            plm_tokenizer = ElectraTokenizer.from_pretrained("token_utils/tokenizer")
            plm_model = PretrainedModel(plm_tokenizer)
            plm_model.to(args.device)

            adapter_model = DpAdapterModel(args, plm_model.config)
            if hasattr(adapter_model, "module"):
                adapter_model.module.load_state_dict(torch.load(""))
            else:
                adapter_model.load_state_dict(torch.load(""))
            adapter_model.to(args.device)

            model = (plm_model, adapter_model)

            result = evaluate(args, model, dev_dataset, mode="test", global_step=global_step)
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