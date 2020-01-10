from typing import Callable, List, Set, Tuple, TypeVar, Optional
# from allennlp.data.dataset_readers.dataset_utils.ontonotes import TypedStringSpan


# 针对CONLL_03
def io_to_bioes(original_tags):
    def _change_prefix(original_tag, new_prefix):
        assert original_tag.find("-") > 0 and len(new_prefix) == 1
        chars = list(original_tag)
        chars[0] = new_prefix
        return "".join(chars)

    def _pop_replace_append(stack, bioes_sequence, new_prefix):
        tag = stack.pop()
        new_tag = _change_prefix(tag, new_prefix)
        bioes_sequence.append(new_tag)

    def _process_stack(stack, bioes_sequence):
        if len(stack) == 1:
            _pop_replace_append(stack, bioes_sequence, "S")
            # _pop_replace_append(stack, bioes_sequence, "U")
        else:
            recoded_stack = []
            _pop_replace_append(stack, recoded_stack, "E")
            # _pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                _pop_replace_append(stack, recoded_stack, "I")
            _pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            bioes_sequence.extend(recoded_stack)

    bioes_sequence = []
    stack = []

    for tag in original_tags:
        if tag == "O":
            if len(stack) == 0:
                bioes_sequence.append(tag)
            else:
                _process_stack(stack, bioes_sequence)
                bioes_sequence.append(tag)
        elif tag[0] == "I":
            if len(stack) == 0:
                stack.append(tag)
            else:
                this_type = tag[2:]
                prev_type = stack[-1][2:]
                if this_type == prev_type:
                    stack.append(tag)
                else:
                    _process_stack(stack, bioes_sequence)
                    stack.append(tag)
        elif tag[0] == 'B':
            if len(stack) > 0:
                _process_stack(stack, bioes_sequence)
            stack.append(tag)
        else:
            raise ValueError("Invalid tag:", tag)

    if len(stack) > 0:
        _process_stack(stack, bioes_sequence)

    return bioes_sequence


def BIEOS2BIO(original_tags):
    def _change_prefix(original_tag, new_prefix):
        assert original_tag.find("-") > 0 and len(new_prefix) == 1
        chars = list(original_tag)
        chars[0] = new_prefix
        return "".join(chars)

    def _pop_replace_append(stack, bio_sequence, new_prefix):
        tag = stack.pop()
        new_tag = _change_prefix(tag, new_prefix)
        bio_sequence.append(new_tag)

    def _process_stack(stack, bio_sequence):
        if len(stack) == 1:
            _pop_replace_append(stack, bio_sequence, "B")
            # _pop_replace_append(stack, bioes_sequence, "U")
        else:
            recoded_stack = []
            _pop_replace_append(stack, recoded_stack, "I")
            # _pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                _pop_replace_append(stack, recoded_stack, "I")
            _pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            bio_sequence.extend(recoded_stack)

    bio_sequence = []
    stack = []

    for tag in original_tags:
        if tag == "O":
            if len(stack) == 0:
                bio_sequence.append(tag)
            else:
                _process_stack(stack, bio_sequence)
                bio_sequence.append(tag)
        elif tag[0] == "E":
            stack.append(tag)
        elif tag[0] == "I":
            stack.append(tag)
        elif tag[0] == "S":
            if len(stack) > 0:
                _process_stack(stack, bio_sequence)
            stack.append(tag)
        elif tag[0] == "B":
            if len(stack) > 0:
                _process_stack(stack, bio_sequence)
            stack.append(tag)
        else:
            raise ValueError("Invalid tag:", tag)

    if len(stack) > 0:
        _process_stack(stack, bio_sequence)

    return bio_sequence


# BIO2BIEOS
def to_bioes(original_tags):
    def _change_prefix(original_tag, new_prefix):
        assert original_tag.find("-") > 0 and len(new_prefix) == 1
        chars = list(original_tag)
        chars[0] = new_prefix
        return "".join(chars)

    def _pop_replace_append(stack, bioes_sequence, new_prefix):
        tag = stack.pop()
        new_tag = _change_prefix(tag, new_prefix)
        bioes_sequence.append(new_tag)

    def _process_stack(stack, bioes_sequence):
        if len(stack) == 1:
            _pop_replace_append(stack, bioes_sequence, "S")
            # _pop_replace_append(stack, bioes_sequence, "U")
        else:
            recoded_stack = []
            _pop_replace_append(stack, recoded_stack, "E")
            # _pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                _pop_replace_append(stack, recoded_stack, "I")
            _pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            bioes_sequence.extend(recoded_stack)

    bioes_sequence = []
    stack = []

    for tag in original_tags:
        if tag == "O":
            if len(stack) == 0:
                bioes_sequence.append(tag)
            else:
                _process_stack(stack, bioes_sequence)
                bioes_sequence.append(tag)
        elif tag[0] == "I":
            if len(stack) == 0:
                stack.append(tag)
            else:
                this_type = tag[2:]
                prev_type = stack[-1][2:]
                if this_type == prev_type:
                    stack.append(tag)
                else:
                    _process_stack(stack, bioes_sequence)
                    stack.append(tag)
        elif tag[0] == "B":
            if len(stack) > 0:
                _process_stack(stack, bioes_sequence)
            stack.append(tag)
        else:
            raise ValueError("Invalid tag:", tag)

    if len(stack) > 0:
        _process_stack(stack, bioes_sequence)

    return bioes_sequence


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return ' '.join(self.tag_sequence)


# bio
# def bio_tags_to_spans(tag_sequence: List[str],
#                       classes_to_ignore: List[str] = None) -> List[TypedStringSpan]:
#     classes_to_ignore = classes_to_ignore or []
#     spans = []
#     span_start = 0
#     span_end = 0
#     active_conll_tag = None
#     for index, string_tag in enumerate(tag_sequence):
#         # Actual BIO tag.
#         bio_tag = string_tag[0]
#         if bio_tag not in ["B", "I", "O"]:
#             raise InvalidTagSequence(tag_sequence)
#         conll_tag = string_tag[2:]
#         if bio_tag == "O" or conll_tag in classes_to_ignore:
#             # The span has ended.
#             if active_conll_tag is not None:
#                 spans.append((span_start, span_end, active_conll_tag))
#             active_conll_tag = None
#             continue
#         elif bio_tag == "B":
#             if active_conll_tag is not None:
#                 spans.append((span_start, span_end, active_conll_tag))
#             active_conll_tag = conll_tag
#             span_start = index
#             span_end = index
#         elif bio_tag == "I" and conll_tag == active_conll_tag:
#             # We're inside a span.
#             span_end += 1
#         else:
#             if active_conll_tag is not None:
#                 spans.append((span_start, span_end, active_conll_tag))
#             active_conll_tag = conll_tag
#             span_start = index
#             span_end = index
#     # Last token might have been a part of a valid span.
#     if active_conll_tag is not None:
#         spans.append((span_start, span_end, active_conll_tag))

#     spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
#     return spans_text


# BIEOS 
def tag_to_spans(tags):
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        entity_name = tag[2:]
        if tag[0] == "U" or tag[0] == "S":
            spans.append((i, i, tag[2:]))
        elif tag[0] == "B":
            start = i
            while tag[0] != "L" and tag[0] != "E":
                i += 1
                if i > len(tags):
                    raise ValueError("Invalid tag sequence: %s" % (" ".join(tags)))
                tag = tags[i]
                if not (tag[0] == "I" or tag[0] == "L" or tag[0] == "E"):
                    raise ValueError("Invalid tag sequence: %s" % (" ".join(tags)))
                if tag[2:] != entity_name:
                    raise ValueError("Invalid entity name match: %s" % (" ".join(tags)))
            spans.append((start, i, tag[2:]))
        else:
            if tag != "O":
                raise ValueError("Invalid tag sequence: %s" % (" ".join(tags)))
        i += 1
    spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
    return spans_text


# 针对 输出的  BIO 不规则的标签不召回
def compute_f1_crf(tags):
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        entity_name = tag[2:]
        start = i
        if tag not in ['[SEP]', '[CLS]']:
            if tag[0] == 'S':
                spans.append((i, i, tag[2:]))
            elif tag[0] == 'B':
                if start != (len(tags) - 1):
                    while tags[start + 1][2:] == entity_name and tags[start + 1][0] != 'O' and tags[start + 1][
                        0] != 'S' and tags[start + 1][0] != 'B':
                        if tags[start][0] == 'E':
                            break
                        start += 1
                        if start == len(tags) - 1:
                            break
                    spans.append((i, start, entity_name))
                else:
                    spans.append((i, start, entity_name))
        i += (start - i) + 1
    spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
    return spans_text


# 针对 输出的  BIEOS 不规则的标签不召回
def compute_f1_crf_BIEOS(tags):
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        entity_name = tag[2:]
        start = i
        if tag not in ['[SEP]', '[CLS]']:
            if tag[0] == 'S':
                spans.append((i, i, tag[2:]))
            elif tag[0] == 'B':
                if start != (len(tags) - 1):
                    while tags[start + 1][2:] == entity_name and tags[start + 1][0] != 'O' and tags[start + 1][
                        0] != 'S' and tags[start + 1][0] != 'B':
                        if tags[start][0] == 'E':
                            break
                        start += 1
                        if start == len(tags) - 1:
                            break
                    if tags[start][0] == 'E':
                        spans.append((i, start, entity_name))

        i += (start - i) + 1
    spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
    return spans_text


# 针对 预测输出不连续  BIEOS/BIO  不规则的标签也召回
def compute_f1_no_crf_BIEOS(tags):
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        entity_name = tag[2:]
        start = i
        if tag not in ['[SEP]', '[CLS]']:
            if tag[0] == 'S':
                spans.append((i, i, tag[2:]))
            elif tag[0] != 'O':
                if start != (len(tags) - 1):
                    while tags[start + 1][2:] == entity_name and tags[start + 1][0] != 'O' and tags[start + 1][
                        0] != 'S' and tags[start + 1][0] != 'B':
                        if tags[start][0] == 'E':
                            break
                        start += 1
                        if start == len(tags) - 1:
                            break
                    spans.append((i, start, entity_name))
                else:
                    spans.append((i, start, entity_name))
        i += (start - i) + 1
    spans_text = "|".join(["%s,%s %s" % (s[0], s[1], s[2]) for s in spans])
    return spans_text


# test = ['B-PER', 'I-MISC', 'S-BER', 'O', 'B-PER', 'E-PER', 'E-PER', 'B-MISC', 'I-MISC', 'E-PER']
# # test = ['B-PER', 'O', 'B-MISC', 'I-MISC', 'B-MISC']
# print(compute_f1_crf_BIEOS(test))
# span = compute_f1_crf(test)
# print(span)
#
# """
#     计算F1样例
# """ ['B-PER', 'E-PER', 'O', 'B-MISC', 'I-MISC', 'E-MISC', 'O'],  ['B-PER', 'E-PER', 'B-MISC', 'I-MISC', 'I-MISC', 'E-MISC', 'O'],
# y_true = [['B-SSS', 'E-SSS', 'O', 'B-MISC', 'I-MISC', 'E-MISC', 'O'], ['B-PER', 'E-PER', 'B-MISC', 'I-MISC', 'I-MISC',
#                                                                        'S-MISC', 'O']]
# y_pred = [['B-ORG', 'E-ORG', 'O', 'B-MISC', 'I-MISC', 'E-MISC', 'O'], ['B-PER', 'E-PER', 'B-MISC', 'I-MISC', 'I-MISC',
#                                                                        'S-MISC', 'O']]
#
# gold_sentences = [compute_f1_crf_BIEOS(i) for i in y_true]
# pred_sentences = [compute_f1_crf_BIEOS(i) for i in y_pred]
# print(gold_sentences)
# print(pred_sentences)
# from metrics import compute_f1
#
# print(compute_f1(gold_sentences, pred_sentences))
