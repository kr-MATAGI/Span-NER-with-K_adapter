import numpy as np

from definition.klue_dp import KlueDPInputExample, KlueDPInputFeatures, dep_labels, pos_labels
from transformers import ElectraTokenizer

from typing import List

#============================================================
def create_examples(file_path: str):
#============================================================
    print(f"[create_examples] file_path: {file_path}")

    sent_id = -1
    examples = []
    with open(file_path, "r", encoding="utf-8") as src_file:
        for line in src_file:
            line = line.strip()
            if line == "" or line == "\n" or line == "\t":
                continue
            if line.startswith("#"):
                parsed = line.strip().split("\t")
                if len(parsed) != 2:  # metadata line about dataset
                    continue
                else:
                    sent_id += 1
                    text = parsed[1].strip()
                    guid = parsed[0].replace("##", "").strip()
            else:
                token_list = [token.replace("\n", "") for token in line.split("\t")] + ["-", "-"]
                examples.append(
                    KlueDPInputExample(
                        guid=guid,
                        text=text,
                        sent_id=sent_id,
                        token_id=int(token_list[0]),
                        token=token_list[1],
                        pos=token_list[3],
                        head=token_list[4],
                        dep=token_list[5],
                    )
                )

    return examples

#============================================================
def convert_features(examples: List[KlueDPInputExample], max_length: int = 128):
#============================================================
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    pos_label_map = {label: i for i, label in enumerate(pos_labels)}
    dep_label_map = {label: i for i, label in enumerate(dep_labels)}

    SENT_ID = 0

    token_list: List[str] = []
    pos_list: List[str] = []
    head_list: List[int] = []
    dep_list: List[str] = []

    features = []
    for example in examples:
        if SENT_ID != example.sent_id:
            SENT_ID = example.sent_id
            encoded = tokenizer.encode_plus(
                " ".join(token_list),
                None,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

            ids, mask = encoded["input_ids"], encoded["attention_mask"]

            bpe_head_mask = [0]
            bpe_tail_mask = [0]
            head_ids = [-1]
            dep_ids = [-1]
            pos_ids = [-1]  # --> CLS token

            for token, head, dep, pos in zip(token_list, head_list, dep_list, pos_list):
                bpe_len = len(tokenizer.tokenize(token))
                head_token_mask = [1] + [0] * (bpe_len - 1)
                tail_token_mask = [0] * (bpe_len - 1) + [1]
                bpe_head_mask.extend(head_token_mask)
                bpe_tail_mask.extend(tail_token_mask)

                head_mask = [head] + [-1] * (bpe_len - 1)
                head_ids.extend(head_mask)
                dep_mask = [dep_label_map[dep]] + [-1] * (bpe_len - 1)
                dep_ids.extend(dep_mask)
                pos_mask = [pos_label_map[pos]] + [-1] * (bpe_len - 1)
                pos_ids.extend(pos_mask)

            bpe_head_mask.append(0)
            bpe_tail_mask.append(0)
            head_ids.append(-1)
            dep_ids.append(-1)
            pos_ids.append(-1)  # END token
            if len(bpe_head_mask) > max_length:
                bpe_head_mask = bpe_head_mask[:max_length]
                bpe_tail_mask = bpe_tail_mask[:max_length]
                head_ids = head_ids[:max_length]
                dep_ids = dep_ids[:max_length]
                pos_ids = pos_ids[:max_length]

            else:
                bpe_head_mask.extend([0] * (max_length - len(bpe_head_mask)))  # padding by max_len
                bpe_tail_mask.extend([0] * (max_length - len(bpe_tail_mask)))  # padding by max_len
                head_ids.extend([-1] * (max_length - len(head_ids)))  # padding by max_len
                dep_ids.extend([-1] * (max_length - len(dep_ids)))  # padding by max_len
                pos_ids.extend([-1] * (max_length - len(pos_ids)))

            feature = KlueDPInputFeatures(
                guid=example.guid,
                ids=ids,
                mask=mask,
                bpe_head_mask=bpe_head_mask,
                bpe_tail_mask=bpe_tail_mask,
                head_ids=head_ids,
                dep_ids=dep_ids,
                pos_ids=pos_ids,
            )
            features.append(feature)

            token_list = []
            pos_list = []
            head_list = []
            dep_list = []

        token_list.append(example.token)
        pos_list.append(example.pos.split("+")[-1])  # 맨 뒤 pos정보만 사용
        head_list.append(int(example.head))
        dep_list.append(example.dep)

    encoded = tokenizer.encode_plus(
        " ".join(token_list),
        None,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    ids, mask = encoded["input_ids"], encoded["attention_mask"]

    bpe_head_mask = [0]
    bpe_tail_mask = [0]
    head_ids = [-1]
    dep_ids = [-1]
    pos_ids = [-1]  # --> CLS token

    for token, head, dep, pos in zip(token_list, head_list, dep_list, pos_list):
        bpe_len = len(tokenizer.tokenize(token))
        head_token_mask = [1] + [0] * (bpe_len - 1)
        tail_token_mask = [0] * (bpe_len - 1) + [1]
        bpe_head_mask.extend(head_token_mask)
        bpe_tail_mask.extend(tail_token_mask)

        head_mask = [head] + [-1] * (bpe_len - 1)
        head_ids.extend(head_mask)
        dep_mask = [dep_label_map[dep]] + [-1] * (bpe_len - 1)
        dep_ids.extend(dep_mask)
        pos_mask = [pos_label_map[pos]] + [-1] * (bpe_len - 1)
        pos_ids.extend(pos_mask)

    bpe_head_mask.append(0)
    bpe_tail_mask.append(0)
    head_ids.append(-1)
    dep_ids.append(-1)  # END token
    bpe_head_mask.extend([0] * (max_length - len(bpe_head_mask)))  # padding by max_len
    bpe_tail_mask.extend([0] * (max_length - len(bpe_tail_mask)))  # padding by max_len
    head_ids.extend([-1] * (max_length - len(head_ids)))  # padding by max_len
    dep_ids.extend([-1] * (max_length - len(dep_ids)))  # padding by max_len
    pos_ids.extend([-1] * (max_length - len(pos_ids)))

    feature = KlueDPInputFeatures(
        guid=example.guid,
        ids=ids,
        mask=mask,
        bpe_head_mask=bpe_head_mask,
        bpe_tail_mask=bpe_tail_mask,
        head_ids=head_ids,
        dep_ids=dep_ids,
        pos_ids=pos_ids,
    )
    features.append(feature)

    for feature in features[:3]:
        print("*** Example ***")
        print("input_ids: %s" % feature.ids)
        print("attention_mask: %s" % feature.mask)
        print("bpe_head_mask: %s" % feature.bpe_head_mask)
        print("bpe_tail_mask: %s" % feature.bpe_tail_mask)
        print("head_id: %s" % feature.head_ids)
        print("dep_ids: %s" % feature.dep_ids)
        print("pos_ids: %s" % feature.pos_ids)

    return features

#============================================================
def create_dataset(file_path: str = "", save_path: str = "", mode: str = ""):
#============================================================
    print(f"[create_dataset] file_path: {file_path}")
    examples = create_examples(file_path=file_path)
    features = convert_features(examples=examples)

    all_input_ids = []
    all_attention_mask = []
    all_bpe_head_mask = []
    all_bpe_tail_mask = []
    all_head_ids = []
    all_dep_ids = []
    all_pos_ids = []
    for feature in features:
        all_input_ids.append(np.array(feature.ids))
        all_attention_mask.append(np.array(feature.mask))
        all_bpe_head_mask.append(np.array(feature.bpe_head_mask))
        all_bpe_tail_mask.append(np.array(feature.bpe_tail_mask))
        all_head_ids.append(np.array(feature.head_ids))
        all_dep_ids.append(np.array(feature.dep_ids))
        all_pos_ids.append(np.array(feature.pos_ids))

    print(f"[create_dataset] made dataset size:")
    print(f"all_input_ids: {len(all_input_ids)}, all_attention_mask: {len(all_attention_mask)}")
    print(f"all_bpe_head_mask: {len(all_bpe_head_mask)}, all_bpe_tail_mask: {len(all_bpe_tail_mask)}")
    print(f"all_head_ids: {len(all_head_ids)}, all_dep_ids: {len(all_dep_ids)}, all_pos_ids: {len(all_pos_ids)}")

    # Save
    np.save(save_path+f"/{mode}_input_ids", all_input_ids)
    np.save(save_path+f"/{mode}_attention_mask", all_attention_mask)
    np.save(save_path+f"/{mode}_bpe_head_mask", all_bpe_head_mask)
    np.save(save_path+f"/{mode}_bpe_tail_mask", all_bpe_tail_mask)
    np.save(save_path+f"/{mode}_head_ids", all_head_ids)
    np.save(save_path+f"/{mode}_dep_ids", all_dep_ids)
    np.save(save_path+f"/{mode}_pos_ids", all_pos_ids)

    print("[create_dataset] Save Complete !")

if "__main__" == __name__:
    print(f"[klue_dp_maker] __MAIN__ !")

    origin_train_file_name = "../corpus/klue_dp/klue-dp-v1.1_train.tsv"
    origin_dev_file_name = "../corpus/klue_dp/klue-dp-v1.1_dev.tsv"

    save_path = "../corpus/npy/klue_dp"
    create_dataset(file_path=origin_train_file_name, save_path=save_path, mode="train")
    create_dataset(file_path=origin_dev_file_name, save_path=save_path, mode="dev")