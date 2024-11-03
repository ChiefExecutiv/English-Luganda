import tensorflow as tf
import keras
from keras import layers
import string, re, random
import numpy as np
from rich.console import Console
import time
import sys

# This is my naively hard-coded dataset
text_pairs = [
    ("How are you?", "[start] Oli otya? [end]"),
    ("Where are you going?", "[start] Ogenda wa? [end]"),
    ("A car is better than a bike.", "[start] Emotoka esiinga egari. [end]"),
    ("We are going to school.", "[start] Tugeenda kusomero. [end]"),
    ("We don't have time.", "[start] Teturina budde. [end]"),
    ("I don't like sweet potatoes.", "[start] Saagala Lumonde. [end]"),
    ("What are you doing?", "[start] Okola ki? [end]"),
    ("It is raining.", "[start] Enkuuba etoonya. [end]"),
    ("Hurry up.", "[start] Yanguwako. [end]"),
    ("Tomorrow we are going to the village.", "[start] Enkya tugeenda mu kyalo. [end]"),
    ("The teacher is getting married.", "[start] Omusomeesa agenda kufumbirwa."),
    ("The king is laughing.", "[start] Kabaka aseka. [end]"),
    ("I will do it tomorrow.", "[start] Nja kikola enkya. [end]"),
    ("But we can't do it.", "[start] Naye tetusobola kikola. [end]"),
    ("The man is tall.", "[start] Omusajja muwanvu. [end]"),
    ("I will pay you tomorrow." , "[start] Nja kusasula enkya. [end]"),
    ("We don't have food.", "[start] Teturina mere. [end]"),
    ("Come and eat food.", "[start] Jangu olye emere. [end]"),
    ("How are you?", "[start] Oli otya? [end]"),
    ("We can't do it", "[start] Tetusobola kikola. [end]"),
    ("The king.", "[start] Kabaka [end]."),
    ("Food is.", "[start] Emere eri. [end]."),
    ("Everyday he comes.", "[start] Buli lunaku ajja [end]"),
    ("What is that?", "[start] Kiki ekyo? [end]"),
    ("Who is that?", "[start] Ani oyo? [end]"),
    ("That person is wearing red.", "[start] Omuntu oyo ayambade red. [end]"),
    ("Where are you?", "[start] Oli wa? [end]"),
    ("Who can save them?", "[start] Ani asobola obataasa? [end]"),
    ("My name is Benjamin", "[start] Erinya lyange lye Benjamin [end]"),
    ("The teacher saw me", "[start] Omusomeesa yandabye. [end]"),
    ("Where is my shoe?", "[start] Engato yange eli wa? [end]"),
    ("I live in Nansana.", "[start] Nsula Nansana. [end]"),
    ("Thank you very much." "[start] Webare nyo. [end]"),
    ("Good night." "[start] Sula bulungi. [end]"),
    ("I want a new car", "[start] Njagala emotoka empya [end]")
]


val_text_pairs = [
    ("My name is Abigail", "[start] Erinya lyange lye Benjamin [end]"),
    ("Where is my shirt?", "[start] Esaati yange eli wa? [end]"),
    ("The woman is tall", "[start] Omukazi muwanvu [end]"),
    ("Everyday she comes", "[start] Buli lunaku ajja [end]"),
    ("That person is wearing blue", "[start] Omuntu oyo ayambade blue [end]")
]

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
    lowercase, f"[{re.escape(strip_chars)}]", "")

vocab_size = 150
sequence_length = 5

total_words = 0

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

# Training
train_english_texts = [pair[0] for pair in text_pairs]
train_luganda_texts = [pair[1] for pair in text_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_luganda_texts)

# Validation
"""
val_english_texts = [pair[0] for pair in val_text_pairs]
val_luganda_texts = [pair[0] for pair in val_text_pairs]
source_vectorization.adapt(val_english_texts)
target_vectorization.adapt(val_luganda_texts)
"""

batch_size = 18
def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({
    "english": eng,
    "luganda": spa[:, :-1],
    }, spa[:, 1:])

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()

# Train dataset
train_ds = make_dataset(text_pairs)

# Validation dataset
val_ds = make_dataset(val_text_pairs)


for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['luganda'].shape: {inputs['luganda'].shape}")


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
        input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
        input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),]
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
            attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
            proj_input = self.layernorm_1(inputs + attention_output)
            proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
        "embed_dim": self.embed_dim,
        "num_heads": self.num_heads,
        "dense_dim": self.dense_dim,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
        layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
        "embed_dim": self.embed_dim,
        "num_heads": self.num_heads,
        "dense_dim": self.dense_dim,
        })
        return config
    
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
        [tf.expand_dims(batch_size, -1),
        tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)
    
    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
        attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)

embed_dim = 256
dense_dim = 2048
num_heads = 8
vocab_size = 150
sequence_length = 5


encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="luganda")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)

decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="Gava_L.keras",
        monitor="val_loss",
        save_best_only=True
    )
]


# Training
transformer.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

transformer.fit(train_ds, epochs=50, callbacks=callbacks, validation_data=val_ds)


# Inference

spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20


def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
        [decoded_sentence])[:, :-1]
        predictions = transformer(
        [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_pairs = [
    ("How are you?", "[start] Oli otya? [end]"),
    ("The man is tall", "[start] Omusajja muwanvu [end]"),
    ("I live in Kampala", "[start] Nsula Nansana [end]"),
]

"""
test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(len(test_pairs)):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
"""

# Demonstration

console = Console()
def type_out(word, delay=0.1):
    for char in word:
        console.print(char, end='', style="bold magenta underline")
        time.sleep(delay)
    console.print()

type_out("Translate English to Luganda(With Deep Learning)", delay=0.1)

sentence = "oli otya?"
while True:
    command = input("English (type: 'End' to stop): ")
    if command == "End":
        break
    else:
        time.sleep(1)
        console.print(f"Luganda: {decode_sequence(command)}", style="bold cyan")
        time.sleep(1)

