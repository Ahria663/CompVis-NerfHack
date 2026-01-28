from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer
import torch

# 1. Load the processor and the base model
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# 2. Define your training arguments
training_args = TrainingArguments(
    output_dir="./models/nerfhack-detector", # Where to save your new "brain"
    per_device_train_batch_size=4,           # Adjust based on your GPU memory
    num_train_epochs=50,                     # How many times to look at the photos
    fp16=True if torch.cuda.is_available() else False, # Use speed boost if on GPU
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
)

# 3. Initialize the Trainer
# (Note: You'll need to load your prepped photos into a 'dataset' object first)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=my_labeled_dataset,
    tokenizer=processor,
)

trainer.train()