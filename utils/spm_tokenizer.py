import sentencepiece.sentencepiece_model_pb2 as model
import sentencepiece as spm


def spm_tokenizer(metadata):
    tokens = metadata["tokenizer.ggml.tokens"]
    bos = metadata.get("tokenizer.ggml.bos_token_id", None)
    eos = metadata.get("tokenizer.ggml.eos_token_id", None)
    unk = metadata.get("tokenizer.ggml.unk_token_id", None)
    pad = metadata.get("tokenizer.ggml.pad_token_id", None)

    normalizer_spec = model.NormalizerSpec(
        name="identity",
        add_dummy_prefix=True,
        precompiled_charsmap=b"",
        remove_extra_whitespaces=False,
        normalization_rule_tsv=b"",
    )

    trainer_spec = model.TrainerSpec(
        model_type="BPE",
        vocab_size=len(tokens),
        input_format="text",
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        treat_whitespace_as_suffix=False,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        vocabulary_output_piece_score=True,
        byte_fallback=False,
        unk_id=unk.item() if unk is not None else None,
        bos_id=bos.item() if bos is not None else None,
        eos_id=eos.item() if eos is not None else None,
        pad_id=pad.item() if pad is not None else None,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        pretokenization_delimiter="",
    )

    model_spec = model.ModelProto(
        normalizer_spec=normalizer_spec,
        trainer_spec=trainer_spec,
    )

    scores = metadata.get("tokenizer.ggml.scores", None)
    scores = scores.tolist() if scores is not None else None
    token_types = metadata.get("tokenizer.ggml.token_types", None)
    token_types = token_types.tolist() if token_types is not None else None

    for i, token in enumerate(tokens):
        score = scores[i] if scores is not None else 0
        token_type = token_types[i] if token_types is not None else model.ModelProto.SentencePiece.NORMAL
        # print(i, token, score, token_type)
        model_spec.pieces.append(
            model.ModelProto.SentencePiece(piece=token, score=score, type=token_type)
        )
    
    # Add <unk> token in not present. This is must for SentencePiece.
    if unk is None:
        model_spec.pieces.append(
            model.ModelProto.SentencePiece(piece="<unk>", score=0, type=model.ModelProto.SentencePiece.UNKNOWN)
        )

    tokenizer = spm.SentencePieceProcessor(model_proto=model_spec.SerializeToString())

    return tokenizer