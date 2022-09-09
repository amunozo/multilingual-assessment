from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer, Trainer, \
    DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset, Dataset
from itertools import chain
import multiprocessing

def group_texts(examples):
    tokenized_inputs = tokenizer(
       examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

block_size = 128
num_proc = multiprocessing.cpu_count()

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
config = AutoConfig.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_config(config)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

bookcorpus = load_dataset("bookcorpus", split="train[:10]")

print('Tokenizing...')

tokenized_datasets = bookcorpus.map(group_texts, batched=True)#, remove_columns=["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)

training_args = TrainingArguments(
    output_dir="random_models",
    learning_rate=0,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

trainer.save_model("random_models/xlm-roberta-base")
