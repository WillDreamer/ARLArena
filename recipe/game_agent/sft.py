from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTTrainer, SFTConfig
from verl.utils.fs import copy_to_local
from functools import partial
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling


# Load your converted dataset
dataset = load_from_disk("/data1/dannie/projects/ARLArena/checkpoints/sft/sft_dataset")

# Load model and processor
from verl.utils import hf_tokenizer, hf_processor
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
model.partial_pretrain = "Qwen/Qwen3-VL-4B-Instruct"
local_path = copy_to_local(model.partial_pretrain, use_shm=False)
processor = hf_processor(local_path, trust_remote_code=False, use_fast=True)

# Define collate function (as per TRL documentation)
def collate_fn(examples, processor):
    """
    Custom collate function for Qwen3-VL that avoids indexing issues.
    """
    # Extract messages and images
    all_messages = []
    all_images = []
    
    for example in examples:
        # Get messages and images from example
        messages = example["messages"]
        images = example.get("images", [])
        
        all_messages.append(messages)
        all_images.append(images)
    
    # Process with processor
    batch = processor(
        text=all_messages,
        images=all_images,
        return_tensors="pt",
        padding=True
    )
    
    # Create labels
    labels = batch["input_ids"].clone()
    
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Mask vision tokens
    vision_token_ids = [151652, 151653, 151655]
    for token_id in vision_token_ids:
        labels[labels == token_id] = -100
    
    batch["labels"] = labels
    
    return batch

# Configure training
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True,
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
)
# collator = DataCollatorForVisionLanguageModeling(processor)
data_collator = partial(collate_fn, processor=processor)
# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    processing_class=processor,
)

# Train
trainer.train()