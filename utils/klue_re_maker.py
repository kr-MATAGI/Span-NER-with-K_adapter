import json
import os
import pickle
import numpy as np

from typing import List, Tuple

import tag_def
from definition.klue_re import KlueRE_Item, EntityItem

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, ElectraTokenizer

#==================================================================
def parse_re_json(json_path: str = "", save_path: str = ""):
#==================================================================
    print(f"[parse_re_json] json_path: {json_path}")

    ret_only_text = []

    root_data = None
    with open(json_path, "r", encoding="utf-8") as json_file:
        root_data = json.load(json_file)

    # Parsing !
    klue_re_items: List[KlueRE_Item] = []
    for r_idx, root_item in enumerate(root_data):
        ret_only_text.append(root_item["sentence"])

        klue_item = KlueRE_Item(
            idx=r_idx,
            guid=root_item["guid"], sentence=root_item["sentence"],
            label=root_item["label"], source=root_item["source"]
        )
        subj_item = EntityItem(
            word=root_item["subject_entity"]["word"], type=root_item["subject_entity"]["type"],
            start_idx=root_item["subject_entity"]["start_idx"], end_idx=root_item["subject_entity"]["end_idx"]
        )
        obj_item = EntityItem(
            word=root_item["object_entity"]["word"], type=root_item["object_entity"]["type"],
            start_idx=root_item["object_entity"]["start_idx"], end_idx=root_item["object_entity"]["end_idx"]
        )
        klue_item.subj_entity = subj_item
        klue_item.obj_entity = obj_item
        klue_re_items.append(klue_item)
    print(f"Total Klue Item Size: {len(klue_re_items)}")

    # Save pickle
    with open(save_path, mode="wb") as save_file:
        pickle.dump(klue_re_items, save_file)

    # Check
    with open(save_path, mode="rb") as check_file:
        print(f"save_size: {len(pickle.load(check_file))}")

    return ret_only_text

#==================================================================
def make_klue_vocab(corpus_path: str = ""):
#==================================================================
    # Vocab init
    tokenizer = BertWordPieceTokenizer(
        clean_text=True, handle_chinese_chars=True,
        strip_accents=False, lowercase=False, wordpieces_prefix="##"
    )

    re_json_list = os.listdir(corpus_path)
    re_json_list = [corpus_path+"/"+file_name for file_name in re_json_list if "klue" in file_name]
    tokenizer.train(
        files=re_json_list, limit_alphabet=6000, vocab_size=32000
    )

    token_save_path = "./ch-{}-wpm-{}-pretty".format(6000, 32000)
    tokenizer.save(token_save_path, True)

    # 생성된 Vocab 파일 전처리
    vocab_path = "./wpm_vcab_all.txt"
    f = open(vocab_path, mode="w", encoding="utf-8")
    with open(token_save_path, mode="r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
        for item in json_data["model"]["vocab"].keys():
            f.write(item+"\n")

    f.close()
    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
    tokenizer.save_pretrained("./tokenizer")

#==================================================================
def mark_entity_spans(
        text: str,
        subject_range: Tuple[int, int],
        object_range: Tuple[int, int],
):
#==================================================================
    subj_start_marker = "<subj>"
    subj_end_marker = "</subj>"
    obj_start_marker = "<obj>"
    obj_end_marker = "</obj>"

    if subject_range < object_range:
        segments = [
            text[: subject_range[0]],
            subj_start_marker,
            text[subject_range[0]: subject_range[1] + 1],
            subj_end_marker,
            text[subject_range[1] + 1: object_range[0]],
            obj_start_marker,
            text[object_range[0]: object_range[1] + 1],
            obj_end_marker,
            text[object_range[1] + 1:],
        ]
    elif subject_range > object_range:
        segments = [
            text[: object_range[0]],
            obj_start_marker,
            text[object_range[0]: object_range[1] + 1],
            obj_end_marker,
            text[object_range[1] + 1: subject_range[0]],
            subj_start_marker,
            text[subject_range[0]: subject_range[1] + 1],
            subj_end_marker,
            text[subject_range[1] + 1:],
        ]
    else:
        raise ValueError("Entity boundaries overlap.")

    marked_text = "".join(segments)
    return marked_text

#==================================================================
def make_klue_re_npy(src_path: str = "", max_seq_len: int = 128,
                     mode: str = "train", debug_mode: bool = False):
#==================================================================
    # tokenizer = BertTokenizer.from_pretrained("./tokenizer")
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    subj_start_marker = "<subj>"
    subj_end_marker = "</subj>"
    obj_start_marker = "<obj>"
    obj_end_marker = "</obj>"
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            subj_start_marker,
            subj_end_marker,
            obj_start_marker,
            obj_end_marker,
        ]
    })
    tokenizer.save_pretrained("./tokenizer")

    data_list: List[KlueRE_Item] = []
    with open(src_path, mode="rb") as data_file:
        data_list = pickle.load(data_file)
        print("[make_klue_re_npy] data file size: ", len(data_list))

    # make
    npy_dict = {
        "guid": [],
        "tokens": [],
        "sent": [],
        "input_ids": [],
        "label_ids": [],
        "attention_mask": [],
        "token_type_ids": []
    }

    with open("../corpus/klue_re/relation_list.json", "r", encoding="utf-8") as f:
        relation_class = json.load(f)["relations"]
        label_map = {label: i for i, label in enumerate(relation_class)}
        label_ids2tok = {i: label for i, label in enumerate(relation_class)}

    for data_item in data_list:
        marked_text = mark_entity_spans(
            data_item.sentence,
            subject_range=(data_item.subj_entity.start_idx, data_item.subj_entity.end_idx),
            object_range=(data_item.obj_entity.start_idx, data_item.obj_entity.end_idx)
        )
        npy_dict["guid"].append(data_item.guid)
        npy_dict["sent"].append(marked_text)

        text_tokens = tokenizer.tokenize(marked_text)
        text_tokens.insert(0, "[CLS]")
        attention_mask = []
        token_type_ids = [0] * max_seq_len
        if max_seq_len <= len(text_tokens):
            text_tokens = text_tokens[:max_seq_len-1]
            text_tokens.append("[SEP]")

            attention_mask += [1] * max_seq_len
        else:
            text_tokens.append("[SEP]")
            attention_mask += [1] * len(text_tokens)

            attention_mask += [0] * (max_seq_len - len(text_tokens))
            text_tokens += ["[PAD]"] * (max_seq_len - len(text_tokens))

        assert max_seq_len == len(text_tokens), f"text_token.len: {len(text_tokens)}, guid: {data_item.guid}"
        assert max_seq_len == len(attention_mask), f"attn_mask.len: {len(attention_mask)}, guid: {data_item.guid}"
        assert max_seq_len == len(token_type_ids), f"token_type_ids.len: {len(token_type_ids)}, guid: {data_item.guid}"

        npy_dict["tokens"].append(text_tokens)
        npy_dict["input_ids"].append(tokenizer.convert_tokens_to_ids(text_tokens))
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["label_ids"].append(label_map[data_item.label])

    # convert npy
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["label_ids"] = np.array(npy_dict["label_ids"])

    if debug_mode:
        print(f"Debug_mode: {debug_mode}")

        for sent, tok, inp, lab in zip(npy_dict["sent"], npy_dict["tokens"], npy_dict["input_ids"], npy_dict["label_ids"]):
            print("Sent: ", sent)
            print("Tokens: ", tok)
            print("Input_ids: ", inp)
            print("Label_ids: ", label_ids2tok[lab])
            input()
    else:
        # save
        np.save(f"../corpus/npy/klue_re/{mode}_guid", npy_dict["guid"])
        np.save(f"../corpus/npy/klue_re/{mode}_sent", npy_dict["sent"])
        np.save(f"../corpus/npy/klue_re/{mode}_input_ids", npy_dict["input_ids"])
        np.save(f"../corpus/npy/klue_re/{mode}_attention_mask", npy_dict["attention_mask"])
        np.save(f"../corpus/npy/klue_re/{mode}_token_type_ids", npy_dict["token_type_ids"])
        np.save(f"../corpus/npy/klue_re/{mode}_label_ids", npy_dict["label_ids"])

        print(f"[make_klue_re_npy] {mode} npy save complete !")

#==================================================================
def make_klue_re(corpus_dir_path: str = ""):
#==================================================================
    print(f"[make_klue_re] corpus_dir_path: {corpus_dir_path}")

    # Check Files
    re_json_list = os.listdir(corpus_dir_path)
    re_json_list = [file_name for file_name in re_json_list if "klue" in file_name]

    train_json_path = ""
    dev_json_path = ""
    for file_name in re_json_list:
        if "train" in file_name:
            train_json_path = corpus_dir_path + "/" + file_name
        elif "dev" in file_name:
            dev_json_path = corpus_dir_path + "/" + file_name
    print(f"[make_klue_re] train_json_path: {train_json_path}")
    print(f"[make_klue_re] dev_json_path: {dev_json_path}")

    # parse klue-re
    print("[make_klue_re] Parse Train Json !")
    train_only_text = parse_re_json(json_path=train_json_path, save_path="../corpus/pkl/klue_re_train.pkl")
    print("[make_klue_re] Parse Dev Json !")
    dev_only_text = parse_re_json(json_path=dev_json_path, save_path="../corpus/pkl/klue_re_dev.pkl")

    all_only_text = train_only_text + dev_only_text
    print("all_only_text.len: ", len(all_only_text))

    # save
    with open("../corpus/klue_re_sent.txt", mode="w", encoding="utf-8") as only_txt_file:
        only_txt_file.write("\n".join(all_only_text))


### MAIN ###
if "__main__" == __name__:
    print(f"[klue_re_parser][__main__] Start !")

    # Init
    corpus_vocab = "../corpus/klue_re"

    # make vocab
    is_make_voacb = True
    if is_make_voacb:
        make_klue_vocab(corpus_path=corpus_vocab)

    # Parse KLUE-RE
    make_klue_re(corpus_dir_path=corpus_vocab)

    # Make *.npy
    make_klue_re_npy(src_path="../corpus/pkl/klue_re_train.pkl", mode="train", debug_mode=False)
    make_klue_re_npy(src_path="../corpus/pkl/klue_re_dev.pkl", mode="dev", debug_mode=False)