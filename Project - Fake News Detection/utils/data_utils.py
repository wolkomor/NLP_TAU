import torch


def load_embeddings(args, device='cpu'):
    embeddings = torch.randn(len(helper.tok2id) + 1, EMBED_SIZE)
    embeddings[0] = 0.
    for word, vec in load_word_vector_mapping(args.vocab, args.vectors, device).items():
        word = normalize(word)
        if word in helper.tok2id:
            embeddings[helper.tok2id[word]] = vec
    logger.info("Initialized embeddings.")

    return embeddings