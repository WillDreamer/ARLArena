"""
Utility functions to visualize images and messages from MultiTurnSFTDataset
"""
import torch
from typing import Dict, Any, Optional
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def visualize_dataset_item(
    item: Dict[str, Any],
    tokenizer,
    processor=None,
    show_image: bool = True,
    show_text: bool = True,
    max_text_length: int = 500,
):
    """
    Visualize a single item from MultiTurnSFTDataset
    
    Args:
        item: Dictionary returned by dataset.__getitem__()
        tokenizer: Tokenizer used to decode input_ids
        processor: Optional processor (for multimodal models)
        show_image: Whether to display images
        show_text: Whether to print decoded text
        max_text_length: Maximum length of text to display
    """
    # Extract input_ids
    input_ids = item.get("input_ids")
    if input_ids is None:
        print("Error: No input_ids found in item")
        return
    
    # Decode text from input_ids
    if show_text:
        if isinstance(input_ids, torch.Tensor):
            input_ids_list = input_ids.cpu().tolist()
        else:
            input_ids_list = input_ids
        
        # Decode the text
        decoded_text = tokenizer.decode(input_ids_list, skip_special_tokens=False)
        
        # Truncate if too long
        if len(decoded_text) > max_text_length:
            decoded_text = decoded_text[:max_text_length] + "..."
        
        print("=" * 80)
        print("DECODED TEXT:")
        print("=" * 80)
        print(decoded_text)
        print("=" * 80)
        print(f"\nText length: {len(decoded_text)} characters")
        print(f"Token length: {len(input_ids_list)} tokens")
    
    # Extract and display images
    if show_image:
        multi_modal_data = item.get("multi_modal_data", {})
        images = multi_modal_data.get("image", [])
        
        if images:
            print(f"\nFound {len(images)} image(s)")
            
            # Display each image
            num_images = len(images)
            if num_images == 1:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                axs = [ax]
            else:
                cols = min(3, num_images)
                rows = (num_images + cols - 1) // cols
                fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
                axs = axs.flatten() if num_images > 1 else [axs]
            
            for i, img in enumerate(images):
                # Convert to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img_np = img.cpu().numpy()
                elif isinstance(img, Image.Image):
                    img_np = np.array(img)
                else:
                    img_np = np.array(img)
                
                # Handle different image formats
                if img_np.ndim == 4:  # Batch dimension
                    img_np = img_np[0]
                if img_np.ndim == 3:
                    # Check if it's CHW format and convert to HWC
                    if img_np.shape[0] == 3 or img_np.shape[0] == 1:
                        img_np = img_np.transpose(1, 2, 0)
                    # Normalize if needed (assuming values are in [0, 1] or need normalization)
                    if img_np.max() > 1.0:
                        img_np = img_np / 255.0
                    # Clip to [0, 1]
                    img_np = np.clip(img_np, 0, 1)
                
                axs[i].imshow(img_np)
                axs[i].set_title(f"Image {i+1}")
                axs[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_images, len(axs)):
                axs[i].axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo images found in this item")
    
    # Print additional information
    print("\n" + "=" * 80)
    print("ITEM METADATA:")
    print("=" * 80)
    print(f"Keys in item: {list(item.keys())}")
    
    if "attention_mask" in item:
        attention_mask = item["attention_mask"]
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.cpu().numpy()
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Non-padding tokens: {attention_mask.sum()}")
    
    if "loss_mask" in item:
        loss_mask = item["loss_mask"]
        if isinstance(loss_mask, torch.Tensor):
            loss_mask = loss_mask.cpu().numpy()
        print(f"Loss mask shape: {loss_mask.shape}")
        print(f"Tokens with loss: {loss_mask.sum()}")
    
    if "multi_modal_inputs" in item:
        multi_modal_inputs = item["multi_modal_inputs"]
        print(f"Multi-modal inputs keys: {list(multi_modal_inputs.keys())}")
        if "image_grid_thw" in multi_modal_inputs:
            print(f"Image grid THW: {multi_modal_inputs['image_grid_thw']}")


def visualize_messages(
    messages: list,
    show_structure: bool = True,
):
    """
    Visualize the messages structure (before tokenization)
    
    Args:
        messages: List of message dictionaries
        show_structure: Whether to show the structure of each message
    """
    print("=" * 80)
    print("MESSAGES STRUCTURE:")
    print("=" * 80)
    
    for i, msg in enumerate(messages):
        print(f"\nMessage {i+1}:")
        print(f"  Role: {msg.get('role', 'unknown')}")
        content = msg.get('content', '')
        
        if isinstance(content, list):
            print(f"  Content type: multimodal (list with {len(content)} items)")
            for j, item in enumerate(content):
                if isinstance(item, dict):
                    print(f"    Item {j+1}: {item.get('type', 'unknown')}")
                    if item.get('type') == 'text':
                        text = item.get('text', '')
                        if len(text) > 100:
                            text = text[:100] + "..."
                        print(f"      Text: {text}")
                    elif item.get('type') == 'image':
                        print(f"      Image placeholder")
        else:
            print(f"  Content type: text")
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"  Content: {content}")
    
    print("=" * 80)


def visualize_from_dataset(
    dataset,
    index: int,
    tokenizer,
    processor=None,
    show_image: bool = True,
    show_text: bool = True,
    show_messages: bool = False,
):
    """
    Convenience function to visualize an item directly from the dataset
    
    Args:
        dataset: MultiTurnSFTDataset instance
        index: Index of the item to visualize
        tokenizer: Tokenizer used in the dataset
        processor: Optional processor
        show_image: Whether to display images
        show_text: Whether to print decoded text
        show_messages: Whether to show the original messages structure
    """
    # Get the item
    item = dataset[index]
    
    # Optionally show messages structure
    if show_messages and hasattr(dataset, 'messages'):
        messages = dataset.messages[index]
        visualize_messages(messages)
        print("\n")
    
    # Visualize the item
    visualize_dataset_item(
        item,
        tokenizer=tokenizer,
        processor=processor,
        show_image=show_image,
        show_text=show_text,
    )

