import os

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub

import queue
import threading

_global_embed = None
_global_sess = None

def worker_initializer():
    global _global_embed, _global_sess
    print(f"[Worker {os.getpid()}] Loading USE model only once…")
    _global_embed, _global_sess = init_embedder()


def init_embedder():
    print("Loading Universal Sentence Encoder...")
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"  # VERSIONE TF1 COMPATIBILE
    cache_dir = "./tfhub_modules"

    os.environ["TFHUB_CACHE_DIR"] = cache_dir

    try:
        embed = hub.Module(module_url)
    except Exception as e:
        raise RuntimeError("Universal Sentence Encoder not found locally and cannot be downloaded.")

    sess = tf.Session()
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    return embed, sess


def async_generator(iterable, max_prefetch=10):
    q = queue.Queue(max_prefetch)

    def producer():
        for item in iterable:
            q.put(item)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        item = q.get()
        if item is None:
            break
        yield item


def batch(iterable, size=512):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            print(f"[batch] full batch ({size})")
            yield buf
            buf = []
    if buf:
        print(f"[batch] final batch ({len(buf)})")
        yield buf


def embed_news_async(news_texts, embed, sess, batch_size=512):
    embeddings = []

    print("[embed] start embedding…")

    for bi, batch_texts in enumerate(batch(async_generator(news_texts), batch_size)):
        print(f"[embed] batch {bi} → {len(batch_texts)} items")
        emb = sess.run(embed(batch_texts))
        embeddings.append(emb)

    final_emb = np.vstack(embeddings)
    print(f"[embed] done, total embeddings: {final_emb.shape}")

    return final_emb